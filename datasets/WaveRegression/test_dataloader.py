import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Create tensors directly (reshaped to [N, 1] for linear regression)

x = torch.tensor([
    [11.0869], [11.4983], [ 4.8112], [12.0550], [ 4.9065],
    [ 7.5511], [ 3.2242], [ 9.9732], [11.8221], [ 1.6737],
    [11.7445], [ 7.4591], [10.9253], [ 7.1341], [ 9.3129],
    [ 5.3961], [11.1268], [ 7.2119], [ 3.3499], [ 7.8848],
    [ 3.3883], [ 5.5463], [ 3.7312], [10.4513], [ 1.3234],
    [ 3.3866], [ 4.5090], [ 2.5053], [ 6.8762], [ 0.0774]
], dtype=torch.float32)

y = torch.tensor([
    [-1.2275], [-0.8980], [-1.1336], [-0.5765], [-1.0036],
    [ 1.1262], [-0.0506], [-0.5638], [-0.7603], [ 1.0278],
    [-0.8882], [ 1.0226], [-1.0855], [ 0.6918], [ 0.1484],
    [-0.7577], [-0.8529], [ 0.7563], [-0.0623], [ 1.0852],
    [-0.0224], [-0.6196], [-0.4385], [-0.7994], [ 0.9243],
    [-0.3197], [-0.9966], [ 0.6466], [ 0.5645], [ 0.1200]
], dtype=torch.float32)



# 2. Instantiate TensorDataset
test_dataset = TensorDataset(x, y)

# 3. Instantiate the DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
