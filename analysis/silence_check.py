import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import core" works from analysis/

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from core import SR, audio_path, sample_train


def timing(path):
    y, _ = librosa.load(path, sr=SR)
    _, idx = librosa.effects.trim(y, top_db=30)
    return [len(y) / SR, idx[0] / SR, (len(y) - idx[1]) / SR]  # total, lead, trail (sec)


rows = sample_train()
X, labels = [], []
for lab in (1, 0):  # 1 = bonafide, 0 = spoof; first 1000 clips of each
    for utt, _, _ in [r for r in rows if r[2] == lab][:1000]:
        X.append(timing(audio_path("train", utt)))
        labels.append(lab)

X, labels = np.array(X), np.array(labels)
for i, n in enumerate(["duration", "lead_silence", "trail_silence"]):
    print(f"{n}: bonafide={X[labels == 1, i].mean():.2f}s  spoof={X[labels == 0, i].mean():.2f}s")

auc = cross_val_score(RandomForestClassifier(100, random_state=0), X, labels, cv=5, scoring="roc_auc")
print("AUC using ONLY silence/duration features:", auc.mean().round(3))