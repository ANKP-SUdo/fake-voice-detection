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

## Attack-diversity experiment (non-standard protocol, not an official ASVspoof result)

`python -m analysis.attack_diversity --seed N` (N = 0, 1, 2). Extra clips from some unseen attack
types (speaker-disjoint, with matched bonafide) are added to training, and the model is tested on
the other attack types. The control adds the same number of clips from already-known attack types.

Pooled EER on held-out attacks:
| seed | test group | baseline | +control | +new attack types |
|------|------------|----------|----------|-------------------|
| 0 | G2 | 9.94%  | 10.29% | 8.00%  |
| 0 | G1 | 17.81% | 17.69% | 13.00% |
| 1 | G2 | 12.21% | 12.24% | 11.23% |
| 1 | G1 | 13.20% | 14.26% | 10.42% |
| 2 | G2 | 15.84% | 16.55% | 11.60% |
| 2 | G1 | 10.77% | 11.53% | 8.29%  |

New attack types helped in 6/6 tables (about -2.9 points on average); the control did not.
A17 (VC) stays hard (28.6-35.6% EER even with extra attack types).

## Demo

Run locally: `streamlit run app.py`. The sidebar offers the SVM baseline, RawNet2, or both side by side.

RawNet2 needs PyTorch and the checkpoint: download `rawnet2_best.pt` from the
[latest Release](https://github.com/ANKP-SUdo/fake-voice-detection/releases) into `models/`
and run `pip install torch`. Without them the app runs the SVM only.

| Eval clip | SVM | RawNet2 |
|---|---|---|
| Real speech | 1.221, REAL | 9.332, REAL |
| Attack A19 (same algorithm as a training attack) | -0.350, FAKE | -6.095, FAKE |
| Attack A17 (unseen attack) | 0.564, REAL (missed) | -9.823, FAKE |

![real](docs/demo_both_real.jpeg) ![a19](docs/demo_both_a19.jpeg) ![a17](docs/demo_both_a17.jpeg)

Three clips are illustrations, not evidence. The eval-set metrics are in the results table below,
and RawNet2 still misses some A17 clips (A17 EER 10.3%).

## Results (ASVspoof 2019 LA eval, 71,237 trials)

| Model | EER | min t-DCF |
|---|---|---|
| SVM + hand-crafted features | 11.54% | 0.27 |
| RawNet2 (raw waveform, 25 epochs) | 6.66% | 0.162 |

RawNet2 per-attack EER: 10 of 13 attacks between 0.5% and 2.8%, A08 6.9%, and the hardest are A17 (10.3%) and A18 (27.6%).

Caveats:
- The SVM used a smaller training subset, so the gain combines model and data effects.
- Single run, seed 0; checkpoint chosen by dev EER (best dev EER 0.70% at epoch 17).
- Eval clips are cut to their first 64,600 samples (about 4 s).

Reproduce the RawNet2 metrics from the saved scores: `python -m analysis.eval_rawnet2`
(reads `analysis/eval_scores_rawnet2.csv`). The trained checkpoint is not in the repo yet.

## Next
- Add RawNet2 as a second model in the demo.
- Future work: AASIST and self-supervised front-ends, data augmentation, and a second seed.