import shutil, random
from pathlib import Path

ROOT = Path("dataset/raw/LA")
PROTO = ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
AUDIO = ROOT / "ASVspoof2019_LA_train" / "flac"
N_PER_CLASS = 2580

random.seed(42)
items = {"bonafide": [], "spoof": []}
for line in PROTO.read_text().splitlines():
    spk, utt, _, attack, label = line.split()
    items[label].append(utt)

for label, folder in [("bonafide", "real"), ("spoof", "fake")]:
    out = Path("dataset") / folder
    out.mkdir(parents=True, exist_ok=True)
    for utt in random.sample(items[label], min(N_PER_CLASS, len(items[label]))):
        shutil.copy(AUDIO / f"{utt}.flac", out / f"{utt}.flac")
print("done")
