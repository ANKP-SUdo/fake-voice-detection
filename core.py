import random
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------- settings
SR = 16000
MAX_SEC = 4
SEED = 42
N_TRAIN_PER_CLASS = 2580
N_DEV_PER_CLASS = 2000
N_EVAL_BONAFIDE = 2000
N_PER_ATTACK = 300

# attack-diversity experiment (analysis/attack_diversity.py)
N_EXTRA_PER_ATTACK = 430          # about the per-attack count of A01-A06 in the base training set
SEEN_BEFORE = {"A16", "A19"}      # same systems as training attacks A04 / A06, so not "unseen"

LA = Path("dataset/raw/LA")
CM = LA / "ASVspoof2019_LA_cm_protocols"
PROTO = {
    "train": CM / "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": CM / "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": CM / "ASVspoof2019.LA.cm.eval.trl.txt",
}
ASV_EVAL_SCORES = LA / "ASVspoof2019_LA_asv_scores" / \
    "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"

CACHE = Path("cache")
# change this if you change extract(), so old features are not reused
FEATURE_VERSION = "mfcc20-v1"


# ---------------------------------------------------------------- data
def read_protocol(split):
    """Returns rows (utt, attack, label): label 1 = bonafide, 0 = spoof."""
    rows = []
    for line in PROTO[split].read_text().splitlines():
        _, utt, _, attack, key = line.split()
        label = 1 if key == "bonafide" else 0
        rows.append((utt, "bonafide" if label else attack, label))
    return rows


def audio_path(split, utt):
    return LA / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt}.flac"


def sample_train():
    """Same clips as prepare_asvspoof.py: bonafide first, then spoof, one seed."""
    rows = read_protocol("train")
    random.seed(SEED)
    out = []
    for lab in (1, 0):
        pool = [r for r in rows if r[2] == lab]
        out += sorted(random.sample(pool, min(N_TRAIN_PER_CLASS, len(pool))))
    return out


def sample_dev():
    rows = read_protocol("dev")
    random.seed(SEED)
    out = []
    for lab in (1, 0):
        pool = [r for r in rows if r[2] == lab]
        out += random.sample(pool, min(N_DEV_PER_CLASS, len(pool)))
    return out


def sample_eval():
    rows = read_protocol("eval")
    random.seed(SEED)
    bona = [r for r in rows if r[2] == 1]
    out = random.sample(bona, min(N_EVAL_BONAFIDE, len(bona)))
    for a in sorted({r[1] for r in rows if r[2] == 0}):
        pool = [r for r in rows if r[1] == a]
        out += random.sample(pool, min(N_PER_ATTACK, len(pool)))
    return out


def split_attacks(seed):
    """Randomly split the unseen attacks into two groups."""
    atts = sorted({r[1] for r in read_protocol("eval") if r[2] == 0} - SEEN_BEFORE)
    random.Random(seed).shuffle(atts)
    k = len(atts) // 2
    return sorted(atts[:k]), sorted(atts[k:])


def diversity_split(train_atts, test_atts, seed):
    """Speaker-disjoint split of the eval protocol.
    Returns (extra_rows, test_rows): extra = spoofs of train_atts + matching bonafide,
    test = bonafide + spoofs of test_atts, all from the other half of the speakers."""
    rows = read_protocol("eval")
    spk = {}
    for line in PROTO["eval"].read_text().splitlines():
        s, utt = line.split()[:2]
        spk[utt] = s
    ids = sorted(set(spk.values()))
    random.Random(f"{seed}-spk").shuffle(ids)
    extra_spk = set(ids[: len(ids) // 2])
    rows_x = [r for r in rows if spk[r[0]] in extra_spk]
    rows_t = [r for r in rows if spk[r[0]] not in extra_spk]

    rx = random.Random(f"{seed}-extra")
    extra = []
    for a in train_atts:
        pool = [r for r in rows_x if r[1] == a]
        extra += rx.sample(pool, min(N_EXTRA_PER_ATTACK, len(pool)))
    pool = [r for r in rows_x if r[2] == 1]
    extra += rx.sample(pool, min(N_EXTRA_PER_ATTACK * len(train_atts), len(pool)))

    rt = random.Random(f"{seed}-test")
    pool = [r for r in rows_t if r[2] == 1]
    test = rt.sample(pool, min(N_EVAL_BONAFIDE, len(pool)))
    for a in test_atts:
        pool = [r for r in rows_t if r[1] == a]
        test += rt.sample(pool, min(N_PER_ATTACK, len(pool)))
    return extra, test


# ---------------------------------------------------------------- features
def extract(path, trim=False):
    if trim:
        y, _ = librosa.load(path, sr=SR)
        # then keep the first 4 s of what is left
        y, _ = librosa.effects.trim(y, top_db=30)
        y = y[: SR * MAX_SEC]
        y = np.pad(y, (0, max(0, 2048 - len(y))))
    else:
        # first 4 s of the raw file
        y, _ = librosa.load(path, sr=SR, duration=MAX_SEC)
    parts = [
        librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20),
        librosa.feature.spectral_centroid(y=y, sr=SR),
        librosa.feature.spectral_bandwidth(y=y, sr=SR),
        librosa.feature.spectral_rolloff(y=y, sr=SR),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.spectral_contrast(y=y, sr=SR),
    ]
    return np.concatenate([np.concatenate([p.mean(axis=1), p.std(axis=1)]) for p in parts])


def features(paths, trim=False):
    """One cached .npy per utterance, so any new selection of clips reuses what exists."""
    d = CACHE / f"{FEATURE_VERSION}_{'trim' if trim else 'raw'}"
    d.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (d / f"{Path(p).stem}.npy").exists()]
    if todo:
        print(
            f"  extracting {len(todo)} files ({'trim' if trim else 'raw'}) ...")
        out = Parallel(n_jobs=-1)(delayed(extract)(p, trim) for p in todo)
        for p, x in zip(todo, out):
            np.save(d / f"{Path(p).stem}.npy", x)
    return np.array([np.load(d / f"{Path(p).stem}.npy") for p in paths])


def load(rows, split, trim=False):
    X = features([audio_path(split, r[0]) for r in rows], trim)
    y = np.array([r[2] for r in rows])
    attacks = np.array([r[1] for r in rows])
    return X, y, attacks


def make_model():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))


# ---------------------------------------------------------------- metrics
# ASVspoof 2019 LA cost model
P_SPOOF = 0.05
P_TAR = (1 - P_SPOOF) * 0.99
P_NON = (1 - P_SPOOF) * 0.01
C_MISS_ASV, C_FA_ASV = 1, 10
C_MISS_CM, C_FA_CM = 1, 10


def eer(y_true, scores):
    """y_true: 1 = bonafide, 0 = spoof. Higher score = more likely bonafide."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def load_asv_scores(path=ASV_EVAL_SCORES):
    """Lines look like: CM_KEY ASV_KEY SCORE (see README.LA.txt)."""
    tar, non, spoof = [], [], []
    with open(path) as f:
        for line in f:
            _, key, score = line.split()
            {"target": tar, "nontarget": non, "spoof": spoof}[
                key].append(float(score))
    return np.array(tar), np.array(non), np.array(spoof)


def asv_error_rates(tar, non, spoof):
    """ASV operating point = its own EER threshold on target vs nontarget."""
    labels = np.r_[np.ones(len(tar)), np.zeros(len(non))]
    fpr, tpr, thr = roc_curve(labels, np.r_[tar, non])
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    t = thr[i]
    return np.mean(non >= t), np.mean(tar < t), np.mean(spoof < t)


def min_tdcf(bona_cm, spoof_cm, asv=None):
    """CM scores: higher = more likely bonafide."""
    tar, non, spoof = asv if asv is not None else load_asv_scores()
    p_fa_asv, p_miss_asv, p_miss_spoof_asv = asv_error_rates(tar, non, spoof)

    c1 = P_TAR * (C_MISS_CM - C_MISS_ASV * p_miss_asv) - \
        P_NON * C_FA_ASV * p_fa_asv
    c2 = C_FA_CM * P_SPOOF * (1 - p_miss_spoof_asv)
    assert c1 > 0 and c2 > 0, "cost constants invalid for these ASV error rates"

    labels = np.r_[np.ones(len(bona_cm)), np.zeros(len(spoof_cm))]
    fpr, tpr, _ = roc_curve(labels, np.r_[bona_cm, spoof_cm])
    p_miss_cm, p_fa_cm = 1 - tpr, fpr          # bonafide rejected, spoof accepted
    tdcf = (c1 * p_miss_cm + c2 * p_fa_cm) / min(c1, c2)
    return float(tdcf.min())