import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 80)
        nn.init.xavier_normal_(self.hidden1.weight)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(80, 80)
        nn.init.xavier_normal_(self.hidden2.weight)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(80, 80)
        nn.init.xavier_normal_(self.hidden3.weight)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(80, 1)
        nn.init.xavier_normal_(self.hidden4.weight)
        self.act4 = nn.Sigmoid()

    def forward(self, X):
        out = self.hidden1(X)
        out = self.act1(out)
        out = self.hidden2(out)
        out = self.act2(out)
        out = self.hidden3(out)
        out = self.act3(out)
        out = self.hidden4(out)
        out = self.act4(out)
        return out

