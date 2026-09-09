import torch
from torch.utils.data import TensorDataset, DataLoader

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
x = torch.tensor(class0 + class1, dtype=torch.float32)
y = torch.tensor([0] * len(class0) + [1] * len(class1), dtype=torch.long)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
