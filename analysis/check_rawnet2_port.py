from pathlib import Path

import pandas as pd
import soundfile as sf

from rawnet2 import load_model, score_waveform

model = load_model("models/rawnet2_run1/rawnet2_best.pt")
df = pd.read_csv("analysis/eval_scores_rawnet2.csv").set_index("utt_id")

for f in sorted(Path("demo_samples").glob("*.flac")):
    # "A17_LA_E_1425990" -> "LA_E_1425990"
    utt = f.stem.split("_", 1)[1]
    x, sr = sf.read(f, dtype="float32")
    assert sr == 16000
    print(
        f"{f.name:32s} local {score_waveform(model, x):8.3f} | Kaggle {df.loc[utt, 'score']:8.3f}")
