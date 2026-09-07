import base64
import io
import torch
from torch.utils.data import TensorDataset

_datasets = {}

def _load(split="train"):
    if split not in _datasets:
        # 1. Create tensors directly (reshaped to [N, 1] for linear regression)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).view(-1, 1)
        y = torch.tensor([2.2, 4.1, 5.9, 8.3, 9.8], dtype=torch.float32).view(-1, 1)

        # 2. Instantiate TensorDataset
        _datasets[split] = TensorDataset(x, y)

    return _datasets[split]


def preview(index=0, split="train"):
    if split not in ("train", "test"):
        split = "train"
 
    dataset = _load(split)
    with open("./datasets/LinearRegression1/img.png", "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("ascii")
    size = 4
    index = 0 if size == 0 else max(0, min(int(index), size - 1))
 
    input, label = dataset[index]
    input_shape = list(input.shape)
    label_shape = list(label.shape) if hasattr(label, "shape") else []
 
    return {
        "kind": "image",
        "mime": "image/png",
        "data": encoded_string,
        "label": str(int(label)),
        "input_shape": input_shape,
        "label_shape": label_shape,
        "index": index,
        "size": size,
    }
