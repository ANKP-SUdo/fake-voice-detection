import random
from collections import defaultdict
from pathlib import Path

import joblib
import librosa
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tdcf import min_tdcf
from train_baseline import CACHE, DEV_AUDIO, DEV_PROTO, MAX_SEC, N_DEV_PER_CLASS, SR, eer
import eval_baseline as ev


def extract_trim(path):
    y, sr = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y[: SR * MAX_SEC]
    y = np.pad(y, (0, max(0, 2048 - len(y))))
    parts = [
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.spectral_contrast(y=y, sr=sr),
    ]
    return np.concatenate([np.concatenate([p.mean(axis=1), p.std(axis=1)]) for p in parts])


def build_trim(paths, name):
    f = CACHE / f"trim_{name}_{len(paths)}.npy"
    if f.exists():
        return np.load(f)
    X = np.array(Parallel(n_jobs=-1)(delayed(extract_trim)(p) for p in paths))
    np.save(f, X)
    return X


def main():
    # train (same folders as the balanced run)
    real = sorted(Path("dataset/real").glob("*.flac"))
    fake = sorted(Path("dataset/fake").glob("*.flac"))
    y_train = np.array([1] * len(real) + [0] * len(fake))
    print(f"train files: {len(real)} real, {len(fake)} fake")
    X_train = build_trim(real + fake, "train")

    # dev (same sampling as train_baseline.py)
    items = {"bonafide": [], "spoof": []}
    for line in DEV_PROTO.read_text().splitlines():
        _, utt, _, _, label = line.split()
        items[label].append(utt)
    random.seed(42)
    dev_real = random.sample(items["bonafide"], min(N_DEV_PER_CLASS, len(items["bonafide"])))
    dev_fake = random.sample(items["spoof"], min(N_DEV_PER_CLASS, len(items["spoof"])))
    y_dev = np.array([1] * len(dev_real) + [0] * len(dev_fake))
    X_dev = build_trim([DEV_AUDIO / f"{u}.flac" for u in dev_real + dev_fake], "dev")

    # eval (same sampling as eval_baseline.py)
    bona = []
    by_attack = defaultdict(list)
    for line in ev.PROTO.read_text().splitlines():
        _, utt, _, attack, label = line.split()
        if label == "bonafide":
            bona.append(utt)
        else:
            by_attack[attack].append(utt)
    random.seed(42)
    utts, labels, attacks = [], [], []
    for u in random.sample(bona, min(ev.N_BONAFIDE, len(bona))):
        utts.append(u)
        labels.append(1)
        attacks.append("bonafide")
    for a in sorted(by_attack):
        for u in random.sample(by_attack[a], min(ev.N_PER_ATTACK, len(by_attack[a]))):
            utts.append(u)
            labels.append(0)
            attacks.append(a)
    y_eval = np.array(labels)
    attacks = np.array(attacks)
    X_eval = build_trim([ev.AUDIO / f"{u}.flac" for u in utts], "eval")

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    model.fit(X_train, y_train)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/svm_trim.joblib")

    print(f"Dev EER (trimmed): {eer(y_dev, model.decision_function(X_dev)) * 100:.2f}%")
    s = model.decision_function(X_eval)
    print(f"Eval EER (trimmed, all attacks): {eer(y_eval, s) * 100:.2f}%")
    # y == 1 is bonafide, so higher decision_function score should mean more bonafide
    assert s[y_eval == 1].mean() > s[y_eval == 0].mean(), "score direction looks flipped"
    print(f"Eval min t-DCF (trimmed): {min_tdcf(s[y_eval == 1], s[y_eval == 0]):.4f}")
    for a in sorted(by_attack):
        m = (attacks == "bonafide") | (attacks == a)
        print(f"  {a}: {eer(y_eval[m], s[m]) * 100:.2f}%")


if __name__ == "__main__":
    main()