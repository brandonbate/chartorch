import base64
import io
import wave

import torch
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

VIEW = {
    "sample": "audio",
    "train_chart": False,
    "test": "failures",
    "num_labels": 35,
}

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

_datasets = {}


class SpeechCommandsCls(Dataset):
    def __init__(self, subset):
        self.ds = SPEECHCOMMANDS(root="./data", download=True, subset=subset)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        waveform, sample_rate, label, *_ = self.ds[index]
        t = waveform.size(-1)
        if t > N_SAMPLES:
            waveform = waveform[..., :N_SAMPLES]
        elif t < N_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - t))
        return waveform, sample_rate, label


def _load(split="train"):
    if split not in _datasets:
        subset = "testing" if split == "test" else "training"
        _datasets[split] = SpeechCommandsCls(subset)
    return _datasets[split]


def _wav_b64(waveform, sample_rate):
    pcm = waveform.detach().cpu().reshape(-1).clamp(-1, 1)
    samples = (pcm * 32767).to(torch.int16).numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(samples)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview(index=0, split="train"):
    if split not in ("train", "test"):
        split = "train"
    dataset = _load(split)
    size = len(dataset)
    index = 0 if size == 0 else max(0, min(int(index), size - 1))
    waveform, sample_rate, label = dataset[index]
    input_shape = list(waveform.shape)
    return {
        "kind": "audio",
        "mime": "audio/wav",
        "data": _wav_b64(waveform, sample_rate),
        "label": str(label),
        "input_shape": [32,32],
        "label_shape": [],
        "index": index,
        "size": size,
    }
