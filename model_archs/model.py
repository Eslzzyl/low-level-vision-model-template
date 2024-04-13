import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    model = ExampleModel()
    a = torch.randn((1, 3, 128, 128))
    b = torch.randn((1, 3, 128, 128))
    x = model(a, b)
    print(x.shape)
