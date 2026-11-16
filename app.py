import hashlib
import os
import tempfile
import urllib.request
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

# Where Cloud gets the RawNet2 checkpoint (GitHub Release rawnet2-v1)
CKPT_URL = "https://github.com/ANKP-SUdo/fake-voice-detection/releases/download/rawnet2-v1/rawnet2_best.pt"
CKPT_PATH = RAWNET_PATHS[1]
CKPT_SHA256 = "d66eb04428a53387dcd5e3f836f6f77c3e1b2e91ec37b95660f3a34c6410701d"

st.set_page_config(page_title="FakeOut", layout="wide")


@st.cache_resource
def get_svm():
    return joblib.load(SVM_PATH)


@st.cache_resource
def get_rawnet(path):
    from rawnet2 import load_model   # lazy import: torch is only needed for RawNet2
    return load_model(path)


@st.cache_resource(show_spinner="Downloading RawNet2 model (67 MB)...")
def ensure_rawnet2_checkpoint():
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(CKPT_PATH) + ".part"
    urllib.request.urlretrieve(CKPT_URL, tmp)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != CKPT_SHA256:
        os.remove(tmp)
        raise RuntimeError("checkpoint hash mismatch")
    os.replace(tmp, CKPT_PATH)
    return str(CKPT_PATH)


def save_upload(uploaded):
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
    return tmp.name


# ---- Which models can run on this machine? ----
rawnet_path = next((p for p in RAWNET_PATHS if p.exists()), None)
rawnet_note = None
if rawnet_path is None:
    try:
        rawnet_path = Path(ensure_rawnet2_checkpoint())
    except Exception as e:
        rawnet_note = f"RawNet2 download failed: {e}"
if rawnet_path is not None:
    try:
        import torch  # noqa: F401
    except ImportError:
        rawnet_path, rawnet_note = None, "RawNet2 needs PyTorch: pip install torch"

options = ["SVM baseline"] + (["RawNet2", "Both"] if rawnet_path else [])
CAPTIONS = {
    "SVM baseline": "MFCC + spectral features, classic ML",
    "RawNet2": "Deep model on the raw waveform",
    "Both": "Compare the two side by side",
}

with st.sidebar:
    st.markdown("## FakeOut")
    st.caption("Audio deepfake detection · ASVspoof 2019 LA")
    st.divider()

    choice = st.radio("Model", options, captions=[CAPTIONS[o] for o in options])
    if rawnet_note:
        st.warning(rawnet_note)
    use_svm = choice in ("SVM baseline", "Both")
    use_rawnet = choice in ("RawNet2", "Both")

    st.divider()
    st.markdown("**Decision threshold**")
    st.caption("Raise it to be stricter about calling audio real, lower it to be more lenient.")
    thr = {}
    if use_svm:
        thr["SVM baseline"] = st.slider("SVM", -3.0, 3.0, 0.0, 0.1,
                                        help="Score at or above this = real. 0 is the model's own boundary.")
    if use_rawnet:
        thr["RawNet2"] = st.slider("RawNet2", -10.0, 10.0, 0.0, 0.5,
                                   help="Score at or above this = real. 0 is the model's own boundary.")

    st.divider()
    with st.expander("Reported results"):
        st.markdown(
            "| Model | EER | min t-DCF |\n|---|---|---|\n"
            "| SVM | 11.54% | 0.27 |\n| RawNet2 | 6.35% | 0.159 |"
        )
        st.caption("Same 5,900-clip ASVspoof 2019 LA eval sample. Lower is better.")

st.title("FakeOut: audio deepfake detector")
st.write("SVM on hand-crafted features and RawNet2 on the raw waveform, trained on ASVspoof 2019 LA.")

with st.expander("About & limits"):
    st.markdown(
        "- On the same 5,900-clip ASVspoof 2019 eval sample, the SVM scored an EER of about 11.5% "
        "and RawNet2 about 6.4% (6.7% on the full eval set).\n"
        "- Both are weakest on unseen attacks (A17, and especially A18 for RawNet2).\n"
        "- RawNet2 scores only the first ~4 s of a clip.\n"
        "- Both were trained on clean studio-style audio, so phone recordings or noisy audio "
        "may be scored unreliably."
    )

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
            is_real = sc >= thr[name]
            with col.container(border=True):
                st.markdown(f"##### {name}")
                color = "green" if is_real else "red"
                label = "Likely real" if is_real else "Likely fake"
                st.markdown(f"## :{color}[{label}]")
                st.caption(f"Score {sc:.2f}  ·  threshold {thr[name]:.1f}")
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