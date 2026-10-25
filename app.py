import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import extract

MODEL_PATH = Path("models/svm_baseline.joblib")
SR = 16000

st.set_page_config(page_title="FakeOut", layout="wide")


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


def save_upload(uploaded):
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
    return tmp.name


st.title("FakeOut: audio deepfake detector")
st.caption(
    "SVM baseline on hand-crafted audio features, trained on ASVspoof 2019 LA.")

threshold = st.sidebar.slider("Decision threshold", -3.0, 3.0, 0.0, 0.1)
st.sidebar.write(
    "Higher score = more likely real. The default of 0 is the SVM's own boundary.")

uploaded = st.file_uploader(
    "Upload a .wav or .flac file", type=["wav", "flac"])

if uploaded is not None:
    path = save_upload(uploaded)
    try:
        y, _ = librosa.load(path, sr=SR, duration=30)  # for display only
        feats = np.asarray(extract(path), dtype=float).reshape(1, -1)
        score = float(get_model().decision_function(feats)[0])

        verdict = "Likely REAL (bonafide)" if score >= threshold else "Likely FAKE (spoof)"
        c1, c2 = st.columns(2)
        c1.metric("Verdict", verdict)
        c2.metric("Score", f"{score:.3f}")
        st.audio(uploaded)

        t = np.arange(len(y)) / SR
        step = max(1, len(y) // 5000)
        wave = go.Figure(go.Scatter(x=t[::step], y=y[::step], mode="lines"))
        wave.update_layout(title="Waveform", xaxis_title="Time (s)",
                           yaxis_title="Amplitude", height=250)
        st.plotly_chart(wave, use_container_width=True)

        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64), ref=np.max
        )
        times = librosa.frames_to_time(np.arange(mel.shape[1]), sr=SR)
        spec = go.Figure(go.Heatmap(z=mel, x=times, colorscale="Viridis"))
        spec.update_layout(title="Mel spectrogram", xaxis_title="Time (s)", yaxis_title="Mel band", height=300)
        st.plotly_chart(spec, use_container_width=True)
    except Exception as e:
        st.error(f"Could not process this file: {e}")
    finally:
        Path(path).unlink(missing_ok=True)

    st.info(
        "Limits: this model scored an EER of about 11.5% on ASVspoof 2019 eval and is weakest on "
        "unseen attacks (A17, A18). It was trained on clean studio-style audio, and real and fake "
        "clips in this dataset differ in leading silence, so phone recordings or noisy audio may "
        "be scored unreliably."
    )
