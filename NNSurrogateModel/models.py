import numpy as np
from torch.autograd import Variable
import gpytorch
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # the dimension need to be adjusted according to the algorithms
        self.model = nn.Sequential(
            nn.Linear(args.input_dim, args.output_dim),
            nn.Sigmoid()
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        return x
