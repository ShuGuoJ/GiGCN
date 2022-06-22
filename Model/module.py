from torch_geometric import nn as gnn
import torch
from torch import nn
from torch.nn import functional as F


# Internal graph convolution
class SubGcn(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)
        # 1
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        logits = self.classifier(h_avg)
        return logits


# Internal graph convolution feature module
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        return h_avg


# External graph convolution
class GraphNet(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GraphConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)
        # self.gcn_3 = gnn.GraphConv(hidden_size, hidden_size)
        # self.bn_3 = gnn.BatchNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        # x_normalization = graph.x
        # h = F.relu(self.gcn_1(x_normalization, graph.edge_index))
        # h = F.relu(self.gcn_2(h, graph.edge_index))
        x_normalization = self.bn_0(graph.x)
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        # h = self.bn_3(F.relu(self.gcn_3(h, graph.edge_index)))
        # h = F.relu(self.gcn_2(h, graph.edge_index))
        logits = self.classifier(h + x_normalization)
        # logits = self.classifier(h)
        return logits


# External graph convolution feature module
class GraphNetFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GCNConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)

    def forward(self, graph):
        x_normalization = self.bn_0(graph.x)
        # x_normalization = graph.x
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        return x_normalization + h



