from core import diversity_split, eer, load, make_model, sample_train, split_attacks
import numpy as np
import argparse
import sys
from pathlib import Path
# so "import core" works from analysis/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def eers(model, X, y, att):
    s = model.decision_function(X)
    out = {"ALL": eer(y, s)}
    for a in sorted(set(att) - {"bonafide"}):
        m = (att == "bonafide") | (att == a)
        out[a] = eer(y[m], s[m])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Does training on more attack types help on other unseen attacks?")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random attack and speaker split")
    ap.add_argument("--trim", action="store_true", help="trim silence first")
    args = ap.parse_args()

    g1, g2 = split_attacks(args.seed)
    print(f"seed {args.seed} | group 1: {g1} | group 2: {g2}")

    X_base, y_base, _ = load(sample_train(), "train", args.trim)
    base = make_model().fit(X_base, y_base)

    for train_atts, test_atts in [(g1, g2), (g2, g1)]:
        extra, test = diversity_split(train_atts, test_atts, args.seed)
        X_ex, y_ex, _ = load(extra, "eval", args.trim)
        X_te, y_te, att_te = load(test, "eval", args.trim)
        plus = make_model().fit(np.vstack([X_base, X_ex]), np.r_[y_base, y_ex])

        r0, r1 = eers(base, X_te, y_te, att_te), eers(plus, X_te, y_te, att_te)
        print(f"\nextra training attacks: {train_atts}")
        print(
            f"extra clips: {int((y_ex == 0).sum())} spoof + {int((y_ex == 1).sum())} bonafide | test clips: {len(y_te)}")
        print(f"{'held-out':10s}{'baseline':>10s}{'+extra':>10s}")
        for k in r0:
            print(f"{k:10s}{r0[k] * 100:9.2f}%{r1[k] * 100:9.2f}%")


if __name__ == "__main__":
    main()
