import pandas as pd

from core import eer, min_tdcf

df = pd.read_csv("analysis/eval_scores_rawnet2.csv")
y, s = df["label"].to_numpy(), df["score"].to_numpy()

print("clips:", len(df), "| bonafide:", int(y.sum()))
print(f"score range: {s.min():.2f} to {s.max():.2f}")
print(f"EER: {eer(y, s) * 100:.2f}%")
print("min t-DCF:", min_tdcf(s[y == 1], s[y == 0]))

print("\nper-attack EER (that attack's spoof vs all bonafide):")
bona = df[df.label == 1]
for att in sorted(a for a in df.attack.unique() if a != "-"):
    sub = pd.concat([bona, df[df.attack == att]])
    print(
        f"  {att}: {eer(sub['label'].to_numpy(), sub['score'].to_numpy()) * 100:.2f}%")
