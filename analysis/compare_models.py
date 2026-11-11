import joblib
import numpy as np
import pandas as pd

from core import eer, load, min_tdcf, sample_eval

# the exact clips the SVM was scored on
rows = sample_eval()
utts = [r[0] for r in rows]
X, y, att = load(rows, "eval")
y, att = np.asarray(y), np.asarray(att)
s_svm = joblib.load("models/svm_baseline.joblib").decision_function(X)

full = pd.read_csv("analysis/eval_scores_rawnet2.csv")
sub = full.set_index("utt_id").loc[utts]
assert (sub["label"].to_numpy() == y).all(), "labels misaligned"
s_rn = sub["score"].to_numpy()

y_f = full["label"].to_numpy()
s_f = full["score"].to_numpy()
att_f = np.where(full["label"] == 1, "bonafide", full["attack"]).astype(str)


def per_attack(y, s, att):
    # each attack's spoof clips against all bonafide clips in the set
    return {a: eer(y[(att == a) | (y == 1)], s[(att == a) | (y == 1)]) * 100
            for a in sorted(set(att) - {"bonafide"})}


print(
    f"sample: {len(y)} clips ({int(y.sum())} bonafide) | full eval: {len(y_f)} clips")
print(f"EER        SVM {eer(y, s_svm)*100:.2f}% | RawNet2 same sample {eer(y, s_rn)*100:.2f}% "
      f"| RawNet2 full {eer(y_f, s_f)*100:.2f}%")
print(f"min t-DCF  SVM {min_tdcf(s_svm[y == 1], s_svm[y == 0]):.3f} | RawNet2 same sample "
      f"{min_tdcf(s_rn[y == 1], s_rn[y == 0]):.3f} | RawNet2 full {min_tdcf(s_f[y_f == 1], s_f[y_f == 0]):.3f}")

p_svm, p_rn, p_full = per_attack(y, s_svm, att), per_attack(
    y, s_rn, att), per_attack(y_f, s_f, att_f)
print("\nattack |   SVM | RawNet2 (same sample) | RawNet2 (full eval)")
for a in p_svm:
    print(f"  {a}  | {p_svm[a]:5.2f} | {p_rn[a]:21.2f} | {p_full[a]:19.2f}")
