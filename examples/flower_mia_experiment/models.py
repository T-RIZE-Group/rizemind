import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=20):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
