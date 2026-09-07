import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Create tensors directly (reshaped to [N, 1] for linear regression)
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).view(-1, 1)
y = torch.tensor([2.2, 4.1, 5.9, 8.3, 9.8], dtype=torch.float32).view(-1, 1)

# 2. Instantiate TensorDataset
test_dataset = TensorDataset(x, y)

# 3. Instantiate the DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
