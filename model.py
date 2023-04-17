import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing

class SGA(MessagePassing):
    def __init__(self, g):
        super().__init__(node_dim=0, aggr='add')

        self.att = nn.Linear(g.F, g.H, bias=False)
        self.lin = nn.Linear(g.F, g.H*g.C, bias=False)
        self.coef = nn.Parameter(torch.zeros(g.K+1, 1, g.H, 1))

    def forward(self, g):
        a = self.att(g.x)/g.T
        a = self.edge_updater(g.edge_index, a=a)

        x = self.lin(g.x).view(-1, g.H, g.C)
        xs = [x]
        for _ in range(g.K):
            x = self.propagate(g.edge_index, x=x, a=a)
            xs.append(x)

        x = torch.stack(xs)
        x = (self.coef*x).mean((0, 2))
        return x

    def message(self, x_j, a):
        return a.unsqueeze(-1) * x_j

    def edge_update(self, a_j, index):
        return softmax(a_j, index)
