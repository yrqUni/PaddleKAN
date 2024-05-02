import paddle
import paddle.nn as nn
from paddle.io import DataLoader, IterableDataset

from kan import KAN

class RandomDataGenerator(IterableDataset):
    def __init__(self, batch_size, num_samples):
        self.batch_size = batch_size
        self.num_samples = num_samples
    
    def __iter__(self):
        for _ in range(self.num_samples):
            x = paddle.rand([self.batch_size, 2])
            u = x[:, 0]
            v = x[:, 1]
            y = (u + v) / (1 + u * v)
            y = y.unsqueeze(-1)
            yield x, y

def test_mul():
    kan = KAN([2, 2, 1], base_activation=nn.Identity)
    optimizer = paddle.optimizer.LBFGS(parameters=kan.parameters(), learning_rate=1)
    dataloader = DataLoader(RandomDataGenerator(1024, 1000), batch_size=None)

    for i, (x, y) in enumerate(dataloader):
        def closure():
            pred_y = kan(x, update_grid=(i % 20 == 0))
            loss = paddle.nn.functional.mse_loss(pred_y.squeeze(-1), y.squeeze(-1))
            reg_loss = kan.regularization_loss(1, 0)
            total_loss = loss + 1e-5 * reg_loss
            print(f"Iteration {i}: MSE Loss = {loss.numpy()}, Regularization Loss = {reg_loss.numpy()}")
            return total_loss
        
        optimizer.step(closure)
        optimizer.clear_grad()

    for layer in kan.layers:
        print(layer.spline_weight)

test_mul()
