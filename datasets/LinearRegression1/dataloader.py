import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Create tensors directly (reshaped to [N, 1] for linear regression)
x = torch.tensor([4.0, 1.0, -2.0, -5.0], dtype=torch.float32).view(-1, 1)
y = torch.tensor([2.0, 0.0,  1.0, -4.0], dtype=torch.float32).view(-1, 1)

# 2. Instantiate TensorDataset
dataset = TensorDataset(x, y)

# 3. Instantiate the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

