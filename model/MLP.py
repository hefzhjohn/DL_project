import torch.nn as nn


class MLP(nn.Module):

    # define model elements
    def __init__(self, input_dim):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, 100)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(100, 500)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(500, 1)
        self.act3 = nn.ReLU()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
