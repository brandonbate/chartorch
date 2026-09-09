import torch
from torch.utils.data import TensorDataset

VIEW = {
    "sample": "text",
    "train_chart": True,
    "test": "regions2d",
}

CLASS_COLORS = {
    0: "#1a73e8",
    1: "#d93025",
}

_datasets = {}

class0 = [
    [8.0, -1.0],
    [7.0, -1.0],
    [-8.0, 5.0],
    [-6.0, 5.0],
    [-5.0, 3.0],
    [-3.0, 2.0],
    [-1.0, 2.0],
    [1.0, 1.0],
    [2.0, 3.0],
    [4.0, 1.0],
    [0.0, 2.0],
    [5.0, 0.0],
]
class1 = [
    [-8.0, 1.0],
    [-6.0, -1.0],
    [-4.0, -2.0],
    [-2.0, -3.0],
    [-1.0, -3.0],
    [-2.0, -2.0],
    [0.0, -2.0],
    [1.0, -4.0],
    [1.0, -3.0],
    [2.0, -4.0],
    [6.0, -6.0],
    [4.0, -5.0],
]


def _load(split="train"):
    if split not in _datasets:
        x = torch.tensor(class0 + class1, dtype=torch.float32)
        y = torch.tensor([0] * len(class0) + [1] * len(class1), dtype=torch.long)
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
        inp, label = dataset[i]
        xy = inp.reshape(-1)
        cls = int(label.reshape(-1)[0].item())
        points.append({
            "x": float(xy[0].item()),
            "y": float(xy[1].item() if xy.numel() > 1 else 0.0),
            "color": CLASS_COLORS.get(cls, "#333333"),
        })
    return points
