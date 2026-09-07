import torch
from torch.utils.data import TensorDataset

VIEW = {
    "sample": "text",
    "train_chart": True,
    "test": "fit1d",
}

_datasets = {}


def _load(split="train"):
    if split not in _datasets:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).view(-1, 1)
        y = torch.tensor([2.2, 4.1, 5.9, 8.3, 9.8], dtype=torch.float32).view(-1, 1)
        _datasets[split] = TensorDataset(x, y)
    return _datasets[split]


def _fmt(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    if hasattr(t, "numel") and t.numel() == 1:
        return str(t.item())
    if hasattr(t, "tolist"):
        return str(t.tolist())
    return str(t)


def preview(index=0, split="train"):
    if split not in ("train", "test"):
        split = "train"
    dataset = _load(split)
    size = len(dataset)
    index = 0 if size == 0 else max(0, min(int(index), size - 1))
    inp, label = dataset[index]
    input_shape = list(inp.shape)
    label_shape = list(label.shape) if hasattr(label, "shape") else []
    return {
        "kind": "text",
        "text": _fmt(inp),
        "label": _fmt(label),
        "input_shape": input_shape,
        "label_shape": label_shape,
        "index": index,
        "size": size,
    }


def chart(split="train"):
    if split not in ("train", "test"):
        split = "train"
    dataset = _load(split)
    points = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        points.append({
            "x": float(x.reshape(-1)[0].item()),
            "y": float(y.reshape(-1)[0].item()),
        })
    return points
