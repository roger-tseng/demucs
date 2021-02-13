import torch
from dataclasses import dataclass, field

@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None

from demucs.wavesplit import WaveSplit

model = WaveSplit(audio_channels=2, X=10)
saved = torch.load(r"checkpoints/musdb=musdb18 samples=40000 epochs=180 repeat=1 batch_size=4 tasnet=True split_valid=True X=10.th", map_location='cpu')

