import base64
import io
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import TensorDataset

VIEW = {
    "sample": "image",
    "train_chart": False,
    "test": "failures",
}

_datasets = {}

# Load the dataset from the file
data = torch.load('datasets/Letters/letters_dataset.pt', weights_only=True)
x = data['images']
y = data['labels']

# Load the dataset from the file
test_data = torch.load('datasets/Letters/test_letters_dataset.pt', weights_only=True)
test_x = data['images']
test_y = data['labels']


# Create Dataset and DataLoader
dataset = TensorDataset(x, y)
test_dataset = TensorDataset(test_x, test_y)

def _load(split="train"):
    if split not in _datasets:
        if split == "train":
            _datasets[split] = dataset
        else:
            print(split)
            _datasets[split] = test_dataset
    return _datasets[split]


def preview(index=0, split="train"):
    if split not in ("train", "test"):
        split = "train"
    dataset = _load(split)
    size = len(dataset)
    index = 0 if size == 0 else max(0, min(int(index), size - 1))
    image, label = dataset[index]
    input_shape = list(image.shape)
    label_shape = list(label.shape) if hasattr(label, "shape") else []
    buf = io.BytesIO()
    to_pil_image(image).save(buf, format="PNG")
    return {
        "kind": "image",
        "mime": "image/png",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "label": str(int(label)),
        "input_shape": input_shape,
        "label_shape": label_shape,
        "index": index,
        "size": size,
    }
