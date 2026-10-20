import random
import sys
from collections import defaultdict


import joblib
import numpy as np

from tdcf import min_tdcf
from train_baseline import LA, build, eer

PROTO = LA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO = LA / "ASVspoof2019_LA_eval" / "flac"
N_BONAFIDE = 2000
N_PER_ATTACK = 300
DEFAULT_MODEL = "models/svm_baseline_2580.joblib"


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Model: {model_path}")

    bona = []
    by_attack = defaultdict(list)
    for line in PROTO.read_text().splitlines():
        _, utt, _, attack, label = line.split()
        if label == "bonafide":
            bona.append(utt)
        else:
            by_attack[attack].append(utt)

    random.seed(42)
    utts, labels, attacks = [], [], []
    for u in random.sample(bona, min(N_BONAFIDE, len(bona))):
        utts.append(u)
        labels.append(1)
        attacks.append("bonafide")
    for a in sorted(by_attack):
        for u in random.sample(by_attack[a], min(N_PER_ATTACK, len(by_attack[a]))):
            utts.append(u)
            labels.append(0)
            attacks.append(a)

    X = build([AUDIO / f"{u}.flac" for u in utts], "eval")
    model = joblib.load(model_path)
    scores = model.decision_function(X)

    y = np.array(labels)
    attacks = np.array(attacks)
    print(f"Eval EER (all attacks): {eer(y, scores) * 100:.2f}%")
    # y == 1 is bonafide, so higher decision_function score = more bonafide
    print(f"Eval min t-DCF: {min_tdcf(scores[y == 1], scores[y == 0]):.4f}")
    for a in sorted(by_attack):
        m = (attacks == "bonafide") | (attacks == a)
        print(f"  {a}: {eer(y[m], scores[m]) * 100:.2f}%")


if __name__ == "__main__":
    main()
