import random
from pathlib import Path

import joblib
import librosa
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SR = 16000
MAX_SEC = 4
LA = Path("dataset/raw/LA")
DEV_PROTO = LA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
DEV_AUDIO = LA / "ASVspoof2019_LA_dev" / "flac"
N_DEV_PER_CLASS = 2000
CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)


def extract(path):
    y, sr = librosa.load(path, sr=SR, duration=MAX_SEC)
    parts = [
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.spectral_contrast(y=y, sr=sr),
    ]
    return np.concatenate([np.concatenate([p.mean(axis=1), p.std(axis=1)]) for p in parts])


def build(paths, name):
    f = CACHE / f"{name}_{len(paths)}.npy"
    if f.exists():
        return np.load(f)
    X = np.array(Parallel(n_jobs=-1)(delayed(extract)(p) for p in paths))
    np.save(f, X)
    return X


def eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def main():
    # training data: real = 1 (bonafide), fake = 0 (spoof)
    real = sorted(Path("dataset/real").glob("*.flac"))
    fake = sorted(Path("dataset/fake").glob("*.flac"))
    paths = real + fake
    y_train = np.array([1] * len(real) + [0] * len(fake))
    print(f"train files: {len(real)} real, {len(fake)} fake")
    X_train = build(paths, "train")

    # dev data from the official protocol
    items = {"bonafide": [], "spoof": []}
    for line in DEV_PROTO.read_text().splitlines():
        _, utt, _, _, label = line.split()
        items[label].append(utt)
    random.seed(42)
    dev_real = random.sample(items["bonafide"], min(N_DEV_PER_CLASS, len(items["bonafide"])))
    dev_fake = random.sample(items["spoof"], min(N_DEV_PER_CLASS, len(items["spoof"])))
    dev_paths = [DEV_AUDIO / f"{u}.flac" for u in dev_real + dev_fake]
    y_dev = np.array([1] * len(dev_real) + [0] * len(dev_fake))
    print(f"dev files: {len(dev_real)} bonafide, {len(dev_fake)} spoof")
    X_dev = build(dev_paths, "dev")

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    model.fit(X_train, y_train)

    scores = model.decision_function(X_dev)  # higher = more likely bonafide
    print(f"Dev EER: {eer(y_dev, scores) * 100:.2f}%")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/svm_baseline.joblib")
    print("saved models/svm_baseline.joblib")


if __name__ == "__main__":
    main()
