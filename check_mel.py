import torch
import torchaudio

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=160, n_mels=64
)
print(mel(torch.randn(1, 16000)).shape)