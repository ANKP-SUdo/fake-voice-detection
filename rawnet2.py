import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SAMPLES = 64600  # about 4 s at 16 kHz, same as training

class SincConv(nn.Module):
    """Learnable band-pass filters: each filter has a trainable low cutoff and bandwidth."""
    def __init__(self, out_channels=20, kernel_size=1025, sr=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels, self.kernel_size, self.sr = out_channels, kernel_size, sr
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz

        to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
        to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
        mel = np.linspace(to_mel(30), to_mel(sr / 2 - (min_low_hz + min_band_hz)), out_channels + 1)
        hz = to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        n_lin = torch.linspace(0, kernel_size / 2 - 1, steps=kernel_size // 2)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sr)

    def forward(self, x):                      # x: (B, 1, T)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sr / 2)
        band = (high - low)[:, 0]
        left = ((torch.sin(torch.matmul(high, self.n_)) - torch.sin(torch.matmul(low, self.n_)))
                / (self.n_ / 2)) * self.window_
        filt = torch.cat([left, 2 * band.view(-1, 1), torch.flip(left, dims=[1])], dim=1)
        filt = filt / (2 * band[:, None])
        return F.conv1d(x, filt.view(self.out_channels, 1, self.kernel_size))


class FMS(nn.Module):
    """Feature map scaling: a learned per-channel gain and shift from the channel average."""
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c, c)

    def forward(self, x):                      # x: (B, C, T)
        s = torch.sigmoid(self.fc(x.mean(-1))).unsqueeze(-1)
        return x * s + s


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, first=False):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(c_in) if not first else None
        self.bn2 = nn.BatchNorm1d(c_out)
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.shortcut = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.act = nn.LeakyReLU(0.3)
        self.pool = nn.MaxPool1d(3)
        self.fms = FMS(c_out)

    def forward(self, x):
        out = x if self.first else self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return self.fms(self.pool(out + x))


class RawNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConv(20, 1025)
        self.bn0 = nn.BatchNorm1d(20)
        self.act = nn.LeakyReLU(0.3)
        self.blocks = nn.Sequential(
            ResBlock(20, 20, first=True), ResBlock(20, 20),
            ResBlock(20, 128), ResBlock(128, 128), ResBlock(128, 128), ResBlock(128, 128))
        self.bn_gru = nn.BatchNorm1d(128)
        self.gru = nn.GRU(128, 1024, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 2)

    def forward(self, x):                      # x: (B, 64600)
        x = F.max_pool1d(torch.abs(self.sinc(x.unsqueeze(1))), 3)
        x = self.act(self.bn0(x))
        x = self.blocks(x)
        x = self.act(self.bn_gru(x)).transpose(1, 2)   # (B, T', 128)
        x, _ = self.gru(x)
        return self.out(self.fc1(x[:, -1, :]))         # logits: (B, 2)


def load_model(path, device="cpu"):
    model = RawNet2()
    state = torch.load(path, map_location="cpu", weights_only=True)
    # strict by default: a mismatch fails loudly
    model.load_state_dict(state)
    return model.to(device).eval()


def score_waveform(model, x):
    """x: mono float32 samples at 16 kHz. Returns a score; higher = more likely real."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < N_SAMPLES:
        # same as training and eval
        x = np.pad(x, (0, N_SAMPLES - len(x)), mode="wrap")
    else:
        # first 4 s, same as eval
        x = x[:N_SAMPLES]
    with torch.no_grad():
        logits = model(torch.from_numpy(x).unsqueeze(0))
    return float(logits[0, 1] - logits[0, 0])
