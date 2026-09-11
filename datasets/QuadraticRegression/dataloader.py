import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Create tensors directly (reshaped to [N, 1] for linear regression)
x = torch.tensor([-3.0, -2.0, -1.0, 0.0,  0.0, -3.0, -1.0, 1.0, 1.0, 2.0, -4.0, -4.0, -3.0, 3.0, 2.0, -4.0], dtype=torch.float32).view(-1, 1)
y = torch.tensor([-1.0, -2.0, -2.0, 0.0, -1.0,  0.0, -4.0, 2.0, 0.0, 4.0,  3.0,  5.0,  3.0, 4.0, 8.0,  8.0], dtype=torch.float32).view(-1, 1)


# 2. Instantiate TensorDataset
dataset = TensorDataset(x, y)

# 3. Instantiate the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

