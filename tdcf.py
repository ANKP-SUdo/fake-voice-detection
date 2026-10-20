import numpy as np
from sklearn.metrics import roc_curve

ASV_EVAL_SCORES = ("dataset/raw/LA/ASVspoof2019_LA_asv_scores/"
                   "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")

# ASVspoof 2019 LA cost model
P_SPOOF = 0.05
P_TAR = (1 - P_SPOOF) * 0.99
P_NON = (1 - P_SPOOF) * 0.01
C_MISS_ASV, C_FA_ASV = 1, 10
C_MISS_CM, C_FA_CM = 1, 10


def load_asv_scores(path=ASV_EVAL_SCORES):
    """Lines look like: CM_KEY ASV_KEY SCORE (see README.LA.txt)."""
    tar, non, spoof = [], [], []
    with open(path) as f:
        for line in f:
            _, key, score = line.split()
            {"target": tar, "nontarget": non, "spoof": spoof}[key].append(float(score))
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

    c1 = P_TAR * (C_MISS_CM - C_MISS_ASV * p_miss_asv) - P_NON * C_FA_ASV * p_fa_asv
    c2 = C_FA_CM * P_SPOOF * (1 - p_miss_spoof_asv)
    assert c1 > 0 and c2 > 0, "cost constants invalid for these ASV error rates"

    labels = np.r_[np.ones(len(bona_cm)), np.zeros(len(spoof_cm))]
    fpr, tpr, _ = roc_curve(labels, np.r_[bona_cm, spoof_cm])
    p_miss_cm, p_fa_cm = 1 - tpr, fpr          # bonafide rejected, spoof accepted
    tdcf = (c1 * p_miss_cm + c2 * p_fa_cm) / min(c1, c2)
    return float(tdcf.min())