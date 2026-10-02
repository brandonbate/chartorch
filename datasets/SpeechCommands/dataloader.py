import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS

LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]
N_SAMPLES = 16000


class SpeechCommandsCls(Dataset):
    def __init__(self, subset):
        self.ds = SPEECHCOMMANDS(root="./data", download=True, subset=subset)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        waveform, _, label, *_ = self.ds[index]
        t = waveform.size(-1)
        if t > N_SAMPLES:
            waveform = waveform[..., :N_SAMPLES]
        elif t < N_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - t))
        return waveform, LABELS.index(label)


dataset = SpeechCommandsCls("training")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
