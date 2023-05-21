import torch.nn as nn
import torch.nn.functional as F
from STTNS.layers import GraphConvolution    #导入图卷积模块
from torch_geometric.nn import GATv2Conv, GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphCNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout):
        super(GraphCNN, self).__init__()
        self.conv1 = GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = GCNConv(in_channels=hid_c, out_channels=out_c)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # data.x data.edge_index
        # [N, C]   [2 ,E]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
