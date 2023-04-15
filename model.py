import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing

class SGA(MessagePassing):
    def __init__(self, g):
        super().__init__(node_dim=1, aggr='add')

        self.a = nn.Parameter(torch.empty(g.H, g.F, 1))
        self.W = nn.Parameter(torch.empty(g.H, g.F, g.C))
        self.c = nn.Parameter(torch.zeros(g.H, g.K+1, 1, 1))

        for i in range(g.H):
            nn.init.xavier_uniform_(self.a[i])
            nn.init.xavier_uniform_(self.W[i])

    def forward(self, g):
        a = torch.matmul(g.x, self.a)/g.T
        a = self.edge_updater(g.edge_index, a=a)

        x = torch.matmul(g.x, self.W)
        xs = [x]
        for _ in range(g.K):
            x = self.propagate(g.edge_index, x=x, a=a)
            xs.append(x)

        x = torch.stack(xs, 1)
        x = (self.c*x).mean((0, 1))
        return x

    def message(self, x_j, a):
        return a * x_j

    def edge_update(self, a_j, index):
        return softmax(a_j, index, dim=1)
