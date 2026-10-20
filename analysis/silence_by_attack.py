import numpy as np, librosa
from pathlib import Path
from collections import defaultdict

LA = Path("dataset/raw/LA")
PROTO = LA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO = LA / "ASVspoof2019_LA_eval" / "flac"
PER_GROUP = 300

groups = defaultdict(list)
for line in PROTO.read_text().splitlines():
    spk, utt, _, attack, key = line.split()
    g = "bonafide" if key == "bonafide" else attack
    if len(groups[g]) < PER_GROUP:
        groups[g].append(utt)

def lead(path):
    y, _ = librosa.load(path, sr=16000)
    _, idx = librosa.effects.trim(y, top_db=30)
    return idx[0] / 16000

for g in sorted(groups):
    vals = [lead(AUDIO / f"{u}.flac") for u in groups[g]]
    print(f"{g:9s} mean={np.mean(vals):.2f}s  median={np.median(vals):.2f}s")