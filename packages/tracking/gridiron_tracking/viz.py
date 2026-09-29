"""A top-down view of a play, for checking whether the data is any good.

This is stream 2's QA surface, not a product screen. The question it exists to answer
is "does this look like football?" — because a coordinate stream can be numerically
plausible and still have a guard running backwards through his own centre, and no
table of numbers will show you that in under an hour.

Two things it draws that a plain dot plot would not, both because this project has
already been burned by their absence:

*Facing is drawn as a tick, separately from motion.* ``o`` and ``dir`` are different
columns and the whole point of docs/animation.md is what happens when they are
conflated. Seeing a receiver's tick pointing back at the ball while his dot travels
to the corner is the fastest possible check that orientation survived the pipeline.

*The tick is dashed when the facing was inferred.* ``o_source`` is carried all the way
into the drawing, so a page full of dashed ticks tells the viewer at a glance that
nothing here measured orientation — which is exactly the situation with a vision
pipeline and with the 2017 release, and exactly the thing a confident-looking
animation otherwise hides.

The output is one self-contained HTML file with the play embedded, so it can be opened
from disk, published, or handed to someone with no Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import PlayTrack
from .field import NFL, FieldSpec
from .interactions import Interactions, derive

O_SOURCES = ["measured", "from_dir", "from_velocity", "assumed", "authored"]


def payload(
    track: PlayTrack,
    interactions: Interactions | None = None,
    spec: FieldSpec = NFL,
    title: str | None = None,
) -> dict[str, Any]:
    """The play, compacted for the browser.

    Player samples are arrays rather than objects: a play is a few thousand of them
    and the file has to stay small enough to embed.
    """
    ix = interactions if interactions is not None else derive(track)

    roster = {}
    for pid, e in track.roster.items():
        side = ix.sides.team_of(track, pid) if ix.sides else None
        roster[pid] = {
            "name": e.name or e.label or pid,
            "num": e.num,
            "team": e.team,
            "side": side or "unknown",
        }

    frames = []
    for f in track.frames:
        carrier = ix.carrier_at(f.t)
        players = {
            pid: [
                round(p.x, 2), round(p.y, 2), round(p.o, 1), round(p.s, 2),
                O_SOURCES.index(p.o_source) if p.o_source in O_SOURCES else 3,
            ]
            for pid, p in f.players.items()
        }
        frames.append({
            "t": round(f.t, 2),
            "p": players,
            "b": [round(f.ball.x, 2), round(f.ball.y, 2)] if f.ball else None,
            "c": carrier,
            "e": list(f.events),
        })

    timeline: list[dict[str, Any]] = []
    for p in ix.possessions:
        if p.player and p.t_end - p.t_start >= 0.3:
            timeline.append({
                "t0": round(p.t_start, 2), "t1": round(p.t_end, 2),
                "kind": "possession", "who": [p.player],
                "label": f"{roster.get(p.player, {}).get('name', p.player)} has the ball",
            })
    for e in ix.engagements:
        if e.kind == "collision":
            continue
        a = roster.get(e.a, {}).get("name", e.a)
        b = roster.get(e.b, {}).get("name", e.b)
        label = {
            "block": f"{a} — {b}",
            "tackle": f"{a} — {b}",
            "tackle_attempt": f"{a} — {b}",
        }.get(e.kind, f"{a} — {b}")
        timeline.append({
            "t0": round(e.t_start, 2), "t1": round(e.t_end, 2),
            "kind": e.kind, "who": [e.a, e.b], "label": label,
            "down": None if e.t_down is None else round(e.t_down, 2),
        })
    for e in ix.evades:
        timeline.append({
            "t0": round(e.t, 2), "t1": round(e.t + 0.3, 2), "kind": "evade",
            "who": [e.carrier, e.beaten],
            "label": f"{roster.get(e.carrier, {}).get('name', e.carrier)} past "
                     f"{roster.get(e.beaten, {}).get('name', e.beaten)}",
        })
    timeline.sort(key=lambda r: (r["t0"], r["kind"]))

    o_counts: dict[str, int] = {}
    for f in track.frames:
        for p in f.players.values():
            o_counts[p.o_source] = o_counts.get(p.o_source, 0) + 1

    return {
        "title": title or f"Play {track.meta.get('play_id', '')}".strip(),
        "meta": {k: str(v) for k, v in track.meta.items()},
        "field": {
            "length": spec.length, "width": spec.width,
            "goalA": spec.goal_a, "goalB": spec.goal_b,
            "hashes": list(spec.hashes), "level": spec.name,
        },
        "roster": roster,
        "frames": frames,
        "timeline": timeline,
        "sides": ix.sides.to_json() if ix.sides else None,
        "oSources": O_SOURCES,
        "oCounts": o_counts,
        "notes": ix.notes,
    }


def to_html(
    track: PlayTrack,
    interactions: Interactions | None = None,
    spec: FieldSpec = NFL,
    title: str | None = None,
) -> str:
    """A self-contained page. No build step, no dependencies, no server."""
    data = payload(track, interactions, spec=spec, title=title)
    return _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def write_html(
    path: str | Path,
    track: PlayTrack,
    interactions: Interactions | None = None,
    spec: FieldSpec = NFL,
    title: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(track, interactions, spec=spec, title=title))
    return path


_TEMPLATE = r"""<title>Gridiron Play Inspector</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --ground:   #0E1116;
    --panel:    #151B22;
    --turf:     #121A1E;
    --turf-end: #0E161A;
    --chalk:    #35434B;
    --chalk-hi: #4E6068;
    --edge:     #222C34;
    --text:     #E4E7E9;
    --text-dim: #8B979E;
    --text-fade:#5D686E;
    --off:      #F2A93B;
    --def:      #4FA8E8;
    --ball:     #EFE9DC;
    --warn:     #E8734A;
    --font-ui:  "Barlow", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-cond:"Barlow Condensed", "Barlow", Impact, sans-serif;
    --font-mono:"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  }
  :root:not([data-theme="dark"]) {
    --ground:   #F4F2ED;
    --panel:    #FBFAF7;
    --turf:     #E7EAE4;
    --turf-end: #DDE2DB;
    --chalk:    #B3BDB8;
    --chalk-hi: #8C9A94;
    --edge:     #D6D8D2;
    --text:     #1B2026;
    --text-dim: #5C666C;
    --text-fade:#8A9399;
    --off:      #B86A11;
    --def:      #1A6698;
    --ball:     #2A2622;
    --warn:     #C2512A;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground:#0E1116; --panel:#151B22; --turf:#121A1E; --turf-end:#0E161A;
      --chalk:#35434B; --chalk-hi:#4E6068; --edge:#222C34;
      --text:#E4E7E9; --text-dim:#8B979E; --text-fade:#5D686E;
      --off:#F2A93B; --def:#4FA8E8; --ball:#EFE9DC; --warn:#E8734A;
    }
  }
  :root[data-theme="dark"] {
    --ground:#0E1116; --panel:#151B22; --turf:#121A1E; --turf-end:#0E161A;
    --chalk:#35434B; --chalk-hi:#4E6068; --edge:#222C34;
    --text:#E4E7E9; --text-dim:#8B979E; --text-fade:#5D686E;
    --off:#F2A93B; --def:#4FA8E8; --ball:#EFE9DC; --warn:#E8734A;
  }

  body {
    background: var(--ground); color: var(--text);
    font-family: var(--font-ui); font-size: 15px; line-height: 1.5;
    padding-block: 28px; padding-left: 20px; padding-right: 20px;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1120px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }

  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; }
  h1 {
    font-family: var(--font-cond); font-weight: 600; font-size: 30px;
    letter-spacing: .01em; margin: 0; text-wrap: balance;
  }
  .sub { color: var(--text-dim); font-size: 14px; }
  .sub b { color: var(--text); font-weight: 600; }

  .stage {
    background: var(--panel); border: 1px solid var(--edge); border-radius: 4px;
    padding: 12px; display: flex; flex-direction: column; gap: 12px;
  }
  canvas { display: block; width: 100%; height: auto; border-radius: 2px; }

  .transport { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  button {
    font-family: var(--font-cond); font-size: 15px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .08em;
    background: transparent; color: var(--text); border: 1px solid var(--chalk-hi);
    border-radius: 2px; padding: 7px 16px; cursor: pointer; min-width: 84px;
  }
  button:hover { border-color: var(--off); color: var(--off); }
  button:focus-visible { outline: 2px solid var(--off); outline-offset: 2px; }

  input[type=range] {
    flex: 1 1 200px; min-width: 140px; accent-color: var(--off);
    background: transparent; cursor: pointer;
  }
  .clock {
    font-family: var(--font-mono); font-size: 15px; font-variant-numeric: tabular-nums;
    color: var(--text); min-width: 78px; text-align: right;
  }
  .clock small { color: var(--text-fade); }

  .legend {
    display: flex; flex-wrap: wrap; gap: 6px 18px;
    font-family: var(--font-cond); text-transform: uppercase;
    letter-spacing: .07em; font-size: 12.5px; color: var(--text-dim);
  }
  .key { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 10px; height: 10px; border-radius: 50%; }
  .tickline { width: 16px; height: 0; border-top: 2px solid var(--text-dim); }
  .tickline.dash { border-top-style: dashed; }

  .cols { display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; align-items: start; }
  @media (max-width: 760px) { .cols { grid-template-columns: 1fr; } }

  section h2 {
    font-family: var(--font-cond); font-size: 13px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .1em; color: var(--text-dim);
    margin: 0 0 10px; padding-bottom: 7px; border-bottom: 1px solid var(--edge);
  }
  .events { max-height: 340px; overflow-y: auto; display: flex; flex-direction: column; }
  .ev {
    display: grid; grid-template-columns: 62px 84px 1fr; gap: 10px; align-items: baseline;
    padding: 7px 8px; border: 0; border-left: 2px solid transparent;
    background: transparent; text-align: left; width: 100%; cursor: pointer;
    font-family: var(--font-ui); font-size: 14px; color: var(--text-dim);
    min-width: 0;
  }
  .ev:hover { background: color-mix(in srgb, var(--off) 8%, transparent); }
  .ev.on { border-left-color: var(--off); color: var(--text); background: color-mix(in srgb, var(--off) 10%, transparent); }
  .ev .t { font-family: var(--font-mono); font-size: 12.5px; font-variant-numeric: tabular-nums; color: var(--text-fade); }
  .ev.on .t { color: var(--off); }
  .ev .k {
    font-family: var(--font-cond); text-transform: uppercase; letter-spacing: .07em;
    font-size: 12px; color: var(--text-fade);
  }
  .ev .l { overflow-wrap: break-word; min-width: 0; }
  .ev .k.tackle, .ev .k.tackle_attempt { color: var(--warn); }
  .ev .k.possession { color: var(--off); }
  .ev .k.evade { color: var(--def); }

  dl.facts { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 7px 14px; font-size: 14px; }
  dl.facts dt {
    font-family: var(--font-cond); text-transform: uppercase; letter-spacing: .07em;
    font-size: 12px; color: var(--text-fade); padding-top: 2px;
  }
  dl.facts dd { margin: 0; font-family: var(--font-mono); font-size: 13px; }
  dl.facts dd.plain { font-family: var(--font-ui); font-size: 14px; }

  .flag {
    margin-top: 12px; padding: 10px 12px; font-size: 13.5px; line-height: 1.45;
    border-left: 2px solid var(--warn); color: var(--text-dim);
    background: color-mix(in srgb, var(--warn) 7%, transparent);
  }
  .flag b { color: var(--text); font-weight: 600; }
  footer { color: var(--text-fade); font-size: 13px; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <h1 id="ttl">Play</h1>
    <div class="sub" id="sub"></div>
  </header>

  <div class="stage">
    <canvas id="cv" width="1200" height="580" aria-label="Top-down view of the play"></canvas>
    <div class="transport">
      <button id="play" type="button">Play</button>
      <input type="range" id="scrub" min="0" max="0" step="1" value="0" aria-label="Frame">
      <div class="clock"><span id="clk">0.00</span><small>s from snap</small></div>
    </div>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--off)"></span>Offense</span>
      <span class="key"><span class="swatch" style="background:var(--def)"></span>Defense</span>
      <span class="key"><span class="swatch" style="background:var(--ball);outline:1px solid var(--chalk-hi)"></span>Ball</span>
      <span class="key"><span class="tickline"></span>Facing measured</span>
      <span class="key"><span class="tickline dash"></span>Facing inferred</span>
    </div>
  </div>

  <div class="cols">
    <section>
      <h2>What happened</h2>
      <div class="events" id="evs"></div>
    </section>
    <section>
      <h2>Where this came from</h2>
      <dl class="facts" id="facts"></dl>
      <div class="flag" id="flag" hidden></div>
    </section>
  </div>

  <footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const F = DATA.field;

// The field is 120 x 53.3 yards. Everything is drawn in yards and scaled once, so a
// dot at x = 48.2 lands where the data says and not where a magic number says.
const PAD = 26;
function sizeCanvas() {
  const w = cv.clientWidth || 1120;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const aspect = (F.width + 6) / (F.length + 4);
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(w * aspect * dpr);
  cv.style.height = Math.round(w * aspect) + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h: w * aspect };
}
let VIEW = sizeCanvas();
function sx(x) { return PAD + (x / F.length) * (VIEW.w - 2 * PAD); }
function sy(y) { return PAD + (y / F.width) * (VIEW.h - 2 * PAD); }
function yards(n) { return (n / F.length) * (VIEW.w - 2 * PAD); }

function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function drawField() {
  const C = {
    turf: css('--turf'), end: css('--turf-end'), chalk: css('--chalk'),
    hi: css('--chalk-hi'), fade: css('--text-fade'),
  };
  ctx.fillStyle = C.turf;
  ctx.fillRect(sx(0), sy(0), sx(F.length) - sx(0), sy(F.width) - sy(0));
  ctx.fillStyle = C.end;
  ctx.fillRect(sx(0), sy(0), sx(F.goalA) - sx(0), sy(F.width) - sy(0));
  ctx.fillRect(sx(F.goalB), sy(0), sx(F.length) - sx(F.goalB), sy(F.width) - sy(0));

  ctx.lineWidth = 1;
  for (let x = F.goalA; x <= F.goalB + 0.01; x += 5) {
    const major = Math.abs((x - F.goalA) % 10) < 0.01;
    ctx.strokeStyle = major ? C.hi : C.chalk;
    ctx.beginPath(); ctx.moveTo(sx(x), sy(0)); ctx.lineTo(sx(x), sy(F.width)); ctx.stroke();
  }
  // Goal lines and the back of each end zone, brighter than the five-yard lines.
  ctx.strokeStyle = C.hi; ctx.lineWidth = 1.6;
  [0, F.goalA, F.goalB, F.length].forEach(x => {
    ctx.beginPath(); ctx.moveTo(sx(x), sy(0)); ctx.lineTo(sx(x), sy(F.width)); ctx.stroke();
  });
  ctx.strokeRect(sx(0), sy(0), sx(F.length) - sx(0), sy(F.width) - sy(0));

  // Hash marks, at this level's inset — the dimension that actually varies between
  // the NFL, college and high school, and the one that silently ruins a registration.
  ctx.strokeStyle = C.chalk; ctx.lineWidth = 1;
  for (let x = F.goalA + 1; x < F.goalB; x += 1) {
    F.hashes.forEach(hy => {
      ctx.beginPath();
      ctx.moveTo(sx(x), sy(hy) - 3); ctx.lineTo(sx(x), sy(hy) + 3); ctx.stroke();
    });
  }

  ctx.fillStyle = C.fade;
  ctx.font = '600 ' + Math.max(9, Math.round(yards(2.6))) + 'px ' + css('--font-cond');
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let x = F.goalA + 10; x <= F.goalB - 10 + 0.01; x += 10) {
    const n = Math.round(50 - Math.abs(x - (F.length / 2)));
    ctx.fillText(String(n), sx(x), sy(F.width) - yards(3.4));
    ctx.fillText(String(n), sx(x), sy(0) + yards(3.4));
  }
}

let idx = 0, playing = false, raf = null, lastTs = 0;

function drawFrame() {
  VIEW = { w: cv.clientWidth, h: cv.clientHeight };
  ctx.clearRect(0, 0, cv.width, cv.height);
  drawField();

  const fr = DATA.frames[idx];
  if (!fr) return;
  const OFF = css('--off'), DEF = css('--def'), BALL = css('--ball');
  const r = Math.max(3.5, yards(0.95));

  // Active engagement lines first, so dots sit on top of them.
  const live = DATA.timeline.filter(e => e.kind !== 'possession' && fr.t >= e.t0 && fr.t <= e.t1);
  live.forEach(e => {
    const a = fr.p[e.who[0]], b = fr.p[e.who[1]];
    if (!a || !b) return;
    ctx.strokeStyle = e.kind === 'evade' ? DEF : css('--warn');
    ctx.lineWidth = e.kind.startsWith('tackle') ? 2 : 1.2;
    ctx.setLineDash(e.kind === 'evade' ? [4, 4] : []);
    ctx.beginPath(); ctx.moveTo(sx(a[0]), sy(a[1])); ctx.lineTo(sx(b[0]), sy(b[1])); ctx.stroke();
    ctx.setLineDash([]);
  });

  for (const pid in fr.p) {
    const [x, y, o, s, osrc] = fr.p[pid];
    const side = (DATA.roster[pid] || {}).side;
    const col = side === 'defense' ? DEF : OFF;

    // Facing, drawn as a tick and dashed when it was inferred. o is NOT dir: a
    // receiver can travel one way and face another, and that divergence is the
    // single most useful thing to be able to see at a glance.
    const rad = -o * Math.PI / 180;
    ctx.strokeStyle = col; ctx.lineWidth = 1.6;
    ctx.setLineDash(osrc === 0 ? [] : [2.5, 2.5]);
    ctx.beginPath();
    ctx.moveTo(sx(x), sy(y));
    ctx.lineTo(sx(x) + Math.cos(rad) * r * 2.3, sy(y) + Math.sin(rad) * r * 2.3);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(sx(x), sy(y), r, 0, Math.PI * 2); ctx.fill();

    if (pid === fr.c) {
      ctx.strokeStyle = BALL; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx(x), sy(y), r + 3.5, 0, Math.PI * 2); ctx.stroke();
    }
  }

  if (fr.b) {
    ctx.fillStyle = BALL;
    ctx.beginPath();
    ctx.ellipse(sx(fr.b[0]), sy(fr.b[1]), Math.max(2.4, r * 0.62), Math.max(1.7, r * 0.42), 0, 0, Math.PI * 2);
    ctx.fill();
  }

  document.getElementById('clk').textContent = (fr.t >= 0 ? '+' : '') + fr.t.toFixed(2);
  document.getElementById('scrub').value = idx;
  document.querySelectorAll('.ev').forEach(el => {
    const i = +el.dataset.i, e = DATA.timeline[i];
    el.classList.toggle('on', fr.t >= e.t0 && fr.t <= e.t1);
  });
}

function step(ts) {
  if (!playing) return;
  if (ts - lastTs >= 100) {                       // the contract's 10 Hz, in real time
    lastTs = ts;
    idx = (idx + 1) % DATA.frames.length;
    drawFrame();
  }
  raf = requestAnimationFrame(step);
}
function setPlaying(on) {
  playing = on;
  document.getElementById('play').textContent = on ? 'Pause' : 'Play';
  if (on) { lastTs = 0; raf = requestAnimationFrame(step); }
  else if (raf) cancelAnimationFrame(raf);
}

/* ---------------------------------------------------------------- chrome */

document.getElementById('ttl').textContent = DATA.title || 'Play';
const sides = DATA.sides || {};
document.getElementById('sub').innerHTML =
  'offense <b>' + (sides.offense || '?') + '</b> · ' +
  DATA.frames.length + ' frames at 10 Hz · ' +
  Object.keys(DATA.roster).length + ' players · ' + (DATA.field.level || '') + ' field';

const evs = document.getElementById('evs');
DATA.timeline.forEach((e, i) => {
  const b = document.createElement('button');
  b.className = 'ev'; b.dataset.i = i; b.type = 'button';
  const dur = e.t1 - e.t0;
  b.innerHTML =
    '<span class="t">' + (e.t0 >= 0 ? '+' : '') + e.t0.toFixed(1) +
      (dur >= 0.35 ? '–' + e.t1.toFixed(1) : '') + '</span>' +
    '<span class="k ' + e.kind + '">' + e.kind.replace('_', ' ') + '</span>' +
    '<span class="l">' + e.label + (e.down != null ? ' — down at +' + e.down.toFixed(1) + 's' : '') + '</span>';
  b.addEventListener('click', () => {
    setPlaying(false);
    let best = 0, bd = Infinity;
    DATA.frames.forEach((f, k) => { const d = Math.abs(f.t - e.t0); if (d < bd) { bd = d; best = k; } });
    idx = best; drawFrame();
  });
  evs.appendChild(b);
});

const total = Object.values(DATA.oCounts).reduce((a, b) => a + b, 0) || 1;
const measured = (DATA.oCounts.measured || 0) / total;
const facts = [
  ['Coordinates', DATA.meta.source === 'ngs'
    ? 'Measured by league tracking' : 'Estimated from video', 'plain'],
  ['Rate', (DATA.meta.rate || 10) + ' Hz'],
  ['Sides from', (sides.method || 'unknown').replace(/_/g, ' ') +
    ' (' + Math.round((sides.confidence || 0) * 100) + '%)', 'plain'],
  ['Facing measured', Math.round(measured * 100) + '% of samples'],
  ['Gaze', 'not measured — no such data exists', 'plain'],
];
const dl = document.getElementById('facts');
facts.forEach(([k, v, cls]) => {
  const dt = document.createElement('dt'); dt.textContent = k;
  const dd = document.createElement('dd'); dd.textContent = v; if (cls) dd.className = cls;
  dl.append(dt, dd);
});

if (DATA.notes && DATA.notes.length) {
  const f = document.getElementById('flag');
  f.hidden = false;
  f.innerHTML = '<b>Caveats carried from the pipeline</b><br>' +
    DATA.notes.map(n => n.charAt(0).toUpperCase() + n.slice(1)).join('<br>');
}
document.getElementById('foot').textContent =
  'Every dot is one row of the exported play file. A dashed facing tick means orientation was inferred, not measured.';

document.getElementById('scrub').max = DATA.frames.length - 1;
document.getElementById('scrub').addEventListener('input', ev => {
  setPlaying(false); idx = +ev.target.value; drawFrame();
});
document.getElementById('play').addEventListener('click', () => setPlaying(!playing));
window.addEventListener('resize', () => { sizeCanvas(); drawFrame(); });

sizeCanvas();
drawFrame();
setPlaying(true);
</script>
"""
