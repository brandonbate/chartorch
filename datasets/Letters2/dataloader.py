import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset from the file
data = torch.load('../../../datasets/Letters/letters_dataset.pt', weights_only=True)
x = data['images']
y = data['labels']

# Create Dataset and DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
