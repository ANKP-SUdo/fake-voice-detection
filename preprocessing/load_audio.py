"""
load_audio.py

Loads a .wav file, resamples it to 16 kHz mono, computes a log-mel
spectrogram, and plots it.

Usage:
    python load_audio.py path/to/audio.wav
"""

import argparse

import numpy as np
import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt


# 16 kHz is the standard rate for ASVspoof and most speech-anti-spoofing work
TARGET_SAMPLE_RATE = 16000
# 80 mel bins is the conventional default for speech (used by RawNet2/AASIST-style models)
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """
    Load a wav file and return a mono waveform tensor resampled to target_sr.
    """
    # soundfile decodes via libsndfile, which ships its own binaries on pip
    # install (no system ffmpeg needed) and has supported wav/flac/mp3 since
    # libsndfile 1.1 -- this sidesteps torchaudio's newer torchcodec/ffmpeg
    # dependency entirely, which is finicky to set up on Windows.
    # shape: (samples,) or (samples, channels)
    audio_np, sr = sf.read(path, dtype="float32")

    # soundfile's channel axis convention is (samples, channels); torchaudio's
    # is (channels, samples), so normalize to that before converting to a tensor.
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]
    else:
        audio_np = audio_np.T
    waveform = torch.from_numpy(audio_np)

    # Collapse to mono by averaging channels, if the file is stereo.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Only resample if the file's native rate doesn't already match.
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform


def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """
    Compute a log-scaled mel spectrogram from a mono waveform tensor.
    Returns a tensor of shape (1, n_mels, time_frames).
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
    )
    mel_spec = mel_transform(waveform)

    # Raw mel-power values span several orders of magnitude; converting to
    # decibels compresses that range so both plots and downstream models
    # see meaningful contrast instead of a few loud frames drowning
    # everything else out.
    log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype="power")(mel_spec)

    return log_mel_spec


def plot_log_mel_spectrogram(
    log_mel_spec: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    title: str = "Log-Mel Spectrogram",
):
    """
    Plot a log-mel spectrogram tensor of shape (1, n_mels, time_frames).
    """
    spec = log_mel_spec.squeeze(0).numpy()
    duration = spec.shape[1] * hop_length / sample_rate

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, duration, 0, spec.shape[0]],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bin")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Load a wav file and plot its log-mel spectrogram."
    )
    parser.add_argument("audio_path", help="Path to the .wav file")
    args = parser.parse_args()

    waveform = load_audio(args.audio_path)
    log_mel_spec = compute_log_mel_spectrogram(waveform)
    plot_log_mel_spectrogram(
        log_mel_spec, title=f"Log-Mel Spectrogram: {args.audio_path}"
    )


if __name__ == "__main__":
    main()
