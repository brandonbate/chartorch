import base64
import io

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

_datasets = {}


def _load(split="train"):
    if split not in _datasets:
        transform = transforms.ToTensor()
        _datasets[split] = datasets.MNIST(
            root="./data",
            train=split != "test",
            download=True,
            transform=transform,
        )
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
