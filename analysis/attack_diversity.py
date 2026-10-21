"""Run from the project root:  python -m analysis.attack_diversity --seed 0"""
import argparse

import numpy as np

from core import (SEEN_BEFORE, diversity_split, eer, load, make_model,
                  sample_train, split_attacks)

CONTROL = sorted(SEEN_BEFORE)   # A16, A19: eval-style spoofs, but not new attack types


def eers(model, X, y, att):
    s = model.decision_function(X)
    out = {"ALL": eer(y, s)}
    for a in sorted(set(att) - {"bonafide"}):
        m = (att == "bonafide") | (att == a)
        out[a] = eer(y[m], s[m])
    return out


def main():
    ap = argparse.ArgumentParser(description="Does training on more attack types help on other unseen attacks?")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random attack and speaker split")
    ap.add_argument("--trim", action="store_true", help="trim silence first")
    args = ap.parse_args()

    g1, g2 = split_attacks(args.seed)
    print(f"seed {args.seed} | group 1: {g1} | group 2: {g2}")

    X_base, y_base, _ = load(sample_train(), "train", args.trim)
    base = make_model().fit(X_base, y_base)

    for train_atts, test_atts in [(g1, g2), (g2, g1)]:
        extra, test = diversity_split(train_atts, test_atts, args.seed)
        n_spoof = sum(1 for r in extra if r[2] == 0)
        # control: same number of extra clips, same speakers, but only already-seen attack types
        ctrl, ctrl_test = diversity_split(CONTROL, test_atts, args.seed,
                                          n_per_attack=n_spoof // len(CONTROL))
        assert ctrl_test == test, "control must use the identical test clips"

        X_ex, y_ex, _ = load(extra, "eval", args.trim)
        X_ct, y_ct, _ = load(ctrl, "eval", args.trim)
        X_te, y_te, att_te = load(test, "eval", args.trim)
        plus = make_model().fit(np.vstack([X_base, X_ex]), np.r_[y_base, y_ex])
        ctrl_model = make_model().fit(np.vstack([X_base, X_ct]), np.r_[y_base, y_ct])

        r0 = eers(base, X_te, y_te, att_te)
        rc = eers(ctrl_model, X_te, y_te, att_te)
        r1 = eers(plus, X_te, y_te, att_te)
        print(f"\nnew attack types added: {train_atts}")
        print(f"extra clips: {int((y_ex == 0).sum())} spoof + {int((y_ex == 1).sum())} bonafide "
              f"| control: {int((y_ct == 0).sum())} spoof + {int((y_ct == 1).sum())} bonafide "
              f"| test clips: {len(y_te)}")
        print(f"{'held-out':10s}{'baseline':>10s}{'+control':>10s}{'+new':>10s}")
        for k in r0:
            print(f"{k:10s}{r0[k] * 100:9.2f}%{rc[k] * 100:9.2f}%{r1[k] * 100:9.2f}%")


if __name__ == "__main__":
    main()