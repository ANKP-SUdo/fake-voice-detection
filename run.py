"""Train the SVM baseline and evaluate it. Run from the project root:  python run.py"""
from pathlib import Path

import joblib

from core import (eer, load, make_model, min_tdcf,
                  sample_dev, sample_eval, sample_train)


def main():
    print("=== SVM baseline ===")

    X_tr, y_tr, _ = load(sample_train(), "train")
    X_dev, y_dev, _ = load(sample_dev(), "dev")
    X_ev, y_ev, att_ev = load(sample_eval(), "eval")
    print(f"train: {len(y_tr)} clips | dev: {len(y_dev)} | eval: {len(y_ev)}")

    model = make_model()
    model.fit(X_tr, y_tr)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/svm_baseline.joblib")

    s_dev = model.decision_function(X_dev)   # higher = more likely bonafide
    s_ev = model.decision_function(X_ev)
    print(f"Dev EER:  {eer(y_dev, s_dev) * 100:.2f}%")
    print(f"Eval EER: {eer(y_ev, s_ev) * 100:.2f}%")
    print(f"Eval min t-DCF: {min_tdcf(s_ev[y_ev == 1], s_ev[y_ev == 0]):.4f}")
    for a in sorted(set(att_ev) - {"bonafide"}):
        m = (att_ev == "bonafide") | (att_ev == a)
        print(f"  {a}: {eer(y_ev[m], s_ev[m]) * 100:.2f}%")


if __name__ == "__main__":
    main()