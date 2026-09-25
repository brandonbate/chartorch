from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset

# Load the dataset from the file
data = torch.load('../../../datasets/Letters/test_letters_dataset.pt', weights_only=True)
x = data['images']
y = data['labels']

# Create Dataset and DataLoader
dataset = TensorDataset(x, y)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
