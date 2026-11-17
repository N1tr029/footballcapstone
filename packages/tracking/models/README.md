# Carrier classifier

`carrier.joblib` is the existing locally trained carrier classifier, copied from
the development artifact `data/labels/carrier.joblib` so a fresh API checkout can
use it. It is a scikit-learn/joblib artifact, separate from the downloaded detector,
SAM 2 and Qwen weights. The automatic API resolves this repository path explicitly.

The classifier estimates possession from tracked formations; it does not directly
detect a ball. Visible-flight endpoint association supplies automatic pass events
and records its own evidence separately. Ambiguous jersey numbers and pass
completion remain unknown.

To train a replacement with this package's labelled clips, use
`python -m gridiron_tracking train-carrier --help`. Training data is local and is
not distributed with the model. The automatic reference video and its model-ready
JSON are documented in `docs/video-analysis-api.md`.
