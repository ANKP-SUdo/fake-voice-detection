import numpy as np, librosa
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def feats(path):
    y, sr = librosa.load(path, sr=16000)
    _, idx = librosa.effects.trim(y, top_db=30)
    return [len(y)/sr, idx[0]/sr, (len(y)-idx[1])/sr]  # total, lead, trail (sec)

X, labels = [], []
for label, folder in [(0, "dataset/real"), (1, "dataset/fake")]:
    files = sorted(Path(folder).glob("*.flac"))[:1000]  # use *.wav if your subset is wav
    for f in files:
        X.append(feats(f)); labels.append(label)

X, labels = np.array(X), np.array(labels)
for i, n in enumerate(["duration", "lead_silence", "trail_silence"]):
    print(f"{n}: real={X[labels==0, i].mean():.2f}s  fake={X[labels==1, i].mean():.2f}s")

auc = cross_val_score(RandomForestClassifier(100, random_state=0), X, labels, cv=5, scoring="roc_auc")
print("AUC using ONLY silence/duration features:", auc.mean().round(3))