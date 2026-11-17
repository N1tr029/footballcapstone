"""Detector-seeded SAM 2 video identities, with bounded object batches.

SAM's per-object memory is independent. Splitting its decoder batch keeps memory
attention bounded on Apple Silicon; the image encoder still runs once per frame.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_signature(frames,anchor,boxes,*,fps,weights,cache_dir,reverse=False,stop=None,rate=10):
    import cv2
    weights=Path(weights);step=max(1,round(fps/rate))
    order=sorted((f for f in frames if (f<=anchor if reverse else f>=anchor) and
        (stop is None or (f>=stop if reverse else f<=stop))),reverse=reverse)
    sources=order[::step]
    if order and order[-1]!=sources[-1]:
        sources.append(order[-1])
    height,width=frames[anchor].shape[:2]
    payload={"version":4,"sources":sources,"boxes":boxes,"weights":weights.name,
        "weightsSha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
        "size":[width,height],"reverse":reverse}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode())
    for f in sources:
        digest.update(cv2.resize(frames[f],(96,54)).tobytes())
    target=Path(cache_dir)/f"sam-{digest.hexdigest()[:16]}.json"
    return target,payload,sources


def predictor_class():
    import torch
    from ultralytics.models.sam.predict import SAM2VideoPredictor

    class BoundedPredictor(SAM2VideoPredictor):
        object_batch_size = 1

        def _run_single_frame_inference(self, output_dict, frame_idx, batch_size,
                is_init_cond_frame, point_inputs, mask_inputs, reverse,
                run_mem_encoder, prev_sam_mask_logits=None, inference_state=None):
            state = inference_state or self.inference_state
            if getattr(self, "_encoded_frame", None) != frame_idx:
                self.backbone_out = self.model.forward_image(state["im"])
                self._encoded_frame = frame_idx
            call = super()._run_single_frame_inference
            if batch_size <= self.object_batch_size:
                return call(output_dict,frame_idx,batch_size,is_init_cond_frame,
                    point_inputs,mask_inputs,reverse,run_mem_encoder,
                    prev_sam_mask_logits,inference_state)

            def sliced(value, lo, hi):
                if isinstance(value,torch.Tensor) and value.ndim and value.shape[0] == batch_size:
                    return value[lo:hi]
                if isinstance(value,list):
                    return [sliced(v,lo,hi) for v in value]
                if isinstance(value,dict):
                    return {k:sliced(v,lo,hi) for k,v in value.items()}
                return value

            chunks=[]
            for lo in range(0,batch_size,self.object_batch_size):
                hi=min(batch_size,lo+self.object_batch_size)
                chunks.append(call(sliced(output_dict,lo,hi),frame_idx,hi-lo,
                    is_init_cond_frame,sliced(point_inputs,lo,hi),
                    sliced(mask_inputs,lo,hi),reverse,run_mem_encoder,
                    sliced(prev_sam_mask_logits,lo,hi),inference_state))

            def combined(values):
                if isinstance(values[0],torch.Tensor):
                    return torch.cat(values,dim=0)
                if isinstance(values[0],list):
                    return [combined([v[i] for v in values]) for i in range(len(values[0]))]
                return values[0]
            return {key:combined([c[key] for c in chunks]) for key in chunks[0]}

    return BoundedPredictor


def track_boxes(frames, anchor, boxes, *, fps, weights, cache_dir, reverse=False,
                stop=None, rate=10, log=lambda *a:None, progress=None):
    """Automatically prompted identities. Output IDs stay fixed across blank masks.

    Decoder output is read before Ultralytics filters empty masks, which otherwise
    renumbers every object after a temporarily occluded athlete. Cache keys include
    image content, source frames, prompts, weights and processing version.
    """
    import cv2
    import numpy as np
    import torch

    weights=Path(weights);cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    weights.parent.mkdir(parents=True,exist_ok=True)
    if not weights.is_file():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(weights))
    target,payload,sources=cache_signature(frames,anchor,boxes,fps=fps,weights=weights,
        cache_dir=cache_dir,reverse=reverse,stop=stop,rate=rate)
    if not sources:
        return {}
    ids=list(boxes);height,width=frames[anchor].shape[:2]
    if target.is_file():
        log(f"    identities: cached SAM 2 {'backward' if reverse else 'forward'} pass")
        return {int(f):v for f,v in json.loads(target.read_text())["boxes"].items()}
    small_width=min(960,width);small_height=round(height*small_width/width)
    small_height-=small_height%2;small_width-=small_width%2
    video=target.with_suffix(".mp4");step=max(1,round(fps/rate))
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),fps/step,(small_width,small_height))
    if not writer.isOpened():
        raise RuntimeError("cannot create SAM's temporary inference stream")
    try:
        for f in sources:
            writer.write(cv2.resize(frames[f],(small_width,small_height)))
    finally:
        writer.release()
    prompts=[[b[0]*small_width/width,b[1]*small_height/height,
              b[2]*small_width/width,b[3]*small_height/height] for b in boxes.values()]
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    Predictor=predictor_class()
    predictor=Predictor(overrides={"conf":.01,"task":"segment","mode":"predict","imgsz":1024,
        "model":str(weights),"save":False,"verbose":False,"device":device})
    predictor.object_batch_size=1 if device=="mps" else 4
    output={}
    try:
        for i,result in enumerate(predictor(source=str(video),bboxes=prompts,stream=True)):
            state=predictor.inference_state;storage="cond_frame_outputs" if i==0 else "non_cond_frame_outputs"
            current=state["output_dict"][storage][next(reversed(state["output_dict"][storage]))]
            logits=current["pred_masks"].flatten(0,1).detach().float().cpu().numpy()
            winner=np.argmax(logits,axis=0);measured={}
            for j,mask in enumerate(logits):
                visible=(mask>0)&(winner==j)
                # SAM letterboxes at the top-left. The decoder mask includes the
                # padded rows; using different x/y scales invents shifted feet.
                scale=max(width,height)/mask.shape[1]
                visible[min(mask.shape[0],round(height/scale)):]=False
                visible[:,min(mask.shape[1],round(width/scale)):]=False
                ys,xs=np.where(visible)
                if len(xs)<8:
                    continue
                box=[float(xs.min()*scale),float(ys.min()*scale),
                     float((xs.max()+1)*scale),float((ys.max()+1)*scale)]
                if box[2]-box[0]<8 or box[3]-box[1]<15:
                    continue
                measured[ids[j]]=box
            output[sources[i]]=measured
            if progress:
                progress(i+1,len(sources))
            if i%10==0:
                log(f"    SAM {'backward' if reverse else 'forward'}: {i+1}/{len(sources)} frames")
        target.write_text(json.dumps({"input":payload,"boxes":output}))
    finally:
        video.unlink(missing_ok=True)
        del predictor
        if device=="mps":
            torch.mps.empty_cache()
    return output
