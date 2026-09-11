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
        x = torch.tensor([
            [11.0869], [11.4983], [ 4.8112], [12.0550], [ 4.9065],
            [ 7.5511], [ 3.2242], [ 9.9732], [11.8221], [ 1.6737],
            [11.7445], [ 7.4591], [10.9253], [ 7.1341], [ 9.3129],
            [ 5.3961], [11.1268], [ 7.2119], [ 3.3499], [ 7.8848],
            [ 3.3883], [ 5.5463], [ 3.7312], [10.4513], [ 1.3234],
            [ 3.3866], [ 4.5090], [ 2.5053], [ 6.8762], [ 0.0774]
        ], dtype=torch.float32)

        y = torch.tensor([
            [-1.2275], [-0.8980], [-1.1336], [-0.5765], [-1.0036],
            [ 1.1262], [-0.0506], [-0.5638], [-0.7603], [ 1.0278],
            [-0.8882], [ 1.0226], [-1.0855], [ 0.6918], [ 0.1484],
            [-0.7577], [-0.8529], [ 0.7563], [-0.0623], [ 1.0852],
            [-0.0224], [-0.6196], [-0.4385], [-0.7994], [ 0.9243],
            [-0.3197], [-0.9966], [ 0.6466], [ 0.5645], [ 0.1200]
        ], dtype=torch.float32)

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
