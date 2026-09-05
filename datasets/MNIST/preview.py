import base64
import io

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

_dataset = None


def _load():
    global _dataset
    if _dataset is None:
        transform = transforms.ToTensor()
        _dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return _dataset


def preview(index=0):
    dataset = _load()
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
