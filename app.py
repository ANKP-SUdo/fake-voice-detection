import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import extract

SVM_PATH = Path("models/svm_baseline.joblib")
RAWNET_PATHS = [Path("models/rawnet2_best.pt"), Path("models/rawnet2_run1/rawnet2_best.pt")]
SR = 16000

st.set_page_config(page_title="FakeOut", layout="wide")


@st.cache_resource
def get_svm():
    return joblib.load(SVM_PATH)


@st.cache_resource
def get_rawnet(path):
    from rawnet2 import load_model   # lazy import: torch is only needed for RawNet2
    return load_model(path)


def save_upload(uploaded):
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
    return tmp.name


# ---- Which models can run on this machine? ----
rawnet_path = next((p for p in RAWNET_PATHS if p.exists()), None)
rawnet_note = None
if rawnet_path is None:
    rawnet_note = "RawNet2 checkpoint not found. Download rawnet2_best.pt (see README) into models/."
else:
    try:
        import torch  # noqa: F401
    except ImportError:
        rawnet_path, rawnet_note = None, "RawNet2 needs PyTorch: pip install torch"

options = ["SVM baseline"] + (["RawNet2", "Both"] if rawnet_path else [])
choice = st.sidebar.radio("Model", options)
if rawnet_note:
    st.sidebar.warning(rawnet_note)

use_svm = choice in ("SVM baseline", "Both")
use_rawnet = choice in ("RawNet2", "Both")

thr = {}
if use_svm:
    thr["SVM baseline"] = st.sidebar.slider("SVM threshold", -3.0, 3.0, 0.0, 0.1)
if use_rawnet:
    thr["RawNet2"] = st.sidebar.slider("RawNet2 threshold", -10.0, 10.0, 0.0, 0.5)
st.sidebar.write("Higher score = more likely real. Score scales differ by model; "
                 "0 is each model's own decision boundary.")

st.title("FakeOut: audio deepfake detector")
st.caption("SVM on hand-crafted features and RawNet2 on the raw waveform, trained on ASVspoof 2019 LA.")

uploaded = st.file_uploader("Upload a .wav or .flac file", type=["wav", "flac"])

if uploaded is not None:
    path = save_upload(uploaded)
    try:
        y, _ = librosa.load(path, sr=SR, duration=30)   # mono float32 at 16 kHz

        scores = {}
        if use_svm:
            feats = np.asarray(extract(path), dtype=float).reshape(1, -1)
            scores["SVM baseline"] = float(get_svm().decision_function(feats)[0])
        if use_rawnet:
            from rawnet2 import score_waveform
            scores["RawNet2"] = score_waveform(get_rawnet(str(rawnet_path)), y)

        cols = st.columns(len(scores))
        for col, (name, sc) in zip(cols, scores.items()):
            verdict = "Likely REAL (bonafide)" if sc >= thr[name] else "Likely FAKE (spoof)"
            col.subheader(name)
            col.metric("Verdict", verdict)
            col.metric("Score", f"{sc:.3f}")
        st.audio(uploaded)

        t = np.arange(len(y)) / SR
        step = max(1, len(y) // 5000)
        wave = go.Figure(go.Scatter(x=t[::step], y=y[::step], mode="lines"))
        wave.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", height=250)
        st.plotly_chart(wave, use_container_width=True)

        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64), ref=np.max)
        times = librosa.frames_to_time(np.arange(mel.shape[1]), sr=SR)
        spec = go.Figure(go.Heatmap(z=mel, x=times, colorscale="Viridis"))
        spec.update_layout(title="Mel spectrogram", xaxis_title="Time (s)", yaxis_title="Mel band", height=300)
        st.plotly_chart(spec, use_container_width=True)
    except Exception as e:
        st.error(f"Could not process this file: {e}")
    finally:
        Path(path).unlink(missing_ok=True)

    st.info(
        "Limits: on ASVspoof 2019 eval the SVM scored an EER of about 11.5% and RawNet2 about 6.7%. "
        "Both are weakest on unseen attacks (A17, and especially A18 for RawNet2). RawNet2 scores only "
        "the first ~4 s of a clip. Both were trained on clean studio-style audio, so phone recordings "
        "or noisy audio may be scored unreliably."
    )