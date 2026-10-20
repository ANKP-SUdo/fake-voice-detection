# FakeOut: audio deepfake detection (ASVspoof 2019 LA)

Flow: protocol files -> pick clips -> extract features -> train SVM -> EER + min t-DCF

- `core.py`  sampling, features, model, metrics
- `run.py`   `python run.py` (raw audio) or `python run.py --trim` (silence trimmed)
- `analysis/` one-off checks, e.g. the leading-silence shortcut

Results (eval, sampled: 2000 bonafide + 300 per attack):
| run  | dev EER | eval EER | min t-DCF |
|------|---------|----------|-----------|
| raw  | 0.78%   | 11.54%   | 0.2705    |
| trim | 2.80%   | 29.24%   | 0.5812    |