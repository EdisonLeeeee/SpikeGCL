from spikegcl import neuron
import torch

from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge, mask_feature
import torch.nn.functional as F


def creat_activation_layer(activation):
    if activation is None:
        return torch.nn.Identity()
    elif activation == "relu":
        return torch.nn.ReLU()
    elif activation == "elu":
        return torch.nn.ELU()
    else:
        raise ValueError("Unknown activation")


def creat_snn_layer(
    alpha=2.0,
    surrogate="sigmoid",
    v_threshold=5e-3,
    snn="PLIF",
):
    tau = 1.0

    if snn in ["LIF", "PLIF"]:
        return getattr(neuron, snn)(tau, alpha=alpha, 
                                    surrogate=surrogate, 
                                    v_threshold=v_threshold, 
                                    detach=True, 
                                   )
    elif snn == "IF":
        return neuron.IF(
            alpha=alpha, surrogate=surrogate, v_threshold=v_threshold, detach=True,
        )
    else:
        raise ValueError("Unknown SNN")


class SpikeGCL(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        T: int = 32,
        alpha: float = 2.0,
        surrogate:str="sigmoid",
        v_threshold:float=5e-3,
        snn:str="PLIF",
        reset:str="zero",
        act:str="elu",
        dropedge: float=0.2,
        dropout: float=0.5,
        bn: bool = True,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.snn = creat_snn_layer(
            alpha=alpha,
            surrogate=surrogate,
            v_threshold=v_threshold,
            snn=snn,
        )
        bn = torch.nn.BatchNorm1d if bn else torch.nn.Identity

        in_channels = [
            x.size(0) for x in torch.chunk(torch.ones(in_channels), T)
        ]
        for channel in in_channels:
            self.convs.append(GCNConv(channel, hidden_channels))
            self.bns.append(bn(channel))

        self.shared_bn = bn(hidden_channels)
        self.shared_conv = GCNConv(hidden_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.act = creat_activation_layer(act)
        self.drop_edge = dropedge
        self.T = T
        self.dropout = torch.nn.Dropout(dropout)
        self.reset = reset

    def encode(self, x, edge_index, edge_weight=None):
        chunks = torch.chunk(x, self.T, dim=1)
        xs = []
        for i, x in enumerate(chunks):
            x = self.dropout(x)
            x = self.bns[i](x)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            x = self.dropout(x)
            x = self.shared_bn(x)
            x = self.shared_conv(x, edge_index, edge_weight)
            x = self.snn(x)
            xs.append(x)
        self.snn.reset(self.reset)
        return xs

    def decode(self, spikes):
        xs = []
        for spike in spikes:
            xs.append(self.lin(spike).sum(1))
        return xs

    def forward(self, x, edge_index, edge_weight=None):
        edge_index2, mask2 = dropout_edge(edge_index, p=self.drop_edge)

        if edge_weight is not None:
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight2 = None

        x2 = x[:, torch.randperm(x.size(1))]

        s1 = self.encode(x, edge_index, edge_weight)
        s2 = self.encode(x2, edge_index2, edge_weight2)

        z1 = self.decode(s1)
        z2 = self.decode(s2)
        return z1, z2

    def loss(self, postive, negative, margin=0.0):
        loss = F.margin_ranking_loss(
            postive, negative, target=torch.ones_like(postive), margin=margin
        )
        return loss
