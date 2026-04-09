import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn.functional import cosine_similarity
import itertools

class EdgeModel(nn.Module):
    """Class used to perform the edge update during Neural message passing, Visual Camera Re-Localization using
    Graph Neural Networks and Relative Pose Supervision."""

    def __init__(self, edge_CNN, in_channels, H, W, in_channels_edge):
        super(EdgeModel, self).__init__()
        self.in_channels, self.H, self.W = in_channels, H, W
        self.in_channels_edge = in_channels_edge
        self.edge_CNN = edge_CNN

    def forward(self, source, target, edge_attr):

        source_full = source.view(source.size(0), self.in_channels, self.H,
                                  self.W).contiguous()  # recover the full size
        target_full = target.view(target.size(0), self.in_channels, self.H,
                                  self.W).contiguous()  # recover the full size
        edge_attr = edge_attr.view(edge_attr.size(0), -1, self.H,
                                   self.W).contiguous()  # recover the full size
        out = torch.cat([source_full, target_full, edge_attr], dim=1).contiguous()
        out = self.edge_CNN(out)

        return out.view(out.size(0), -1)


def batch_cosine_similarity(x, y, pooling=None):
    b = x.size(0)
    c = x.size(1)

    if pooling is None:
        outx = x.view(b, c, -1)
        outy = y.view(b, c, -1)

    elif pooling == 'max':
        outx = F.adaptive_max_pool2d(x, 1)
        outy = F.adaptive_max_pool2d(y, 1)

        outx = outx.view(b, c, -1)
        outy = outy.view(b, c, -1)

    elif pooling == 'avg':
        outx = F.adaptive_avg_pool2d(x, 1)
        outy = F.adaptive_avg_pool2d(y, 1)

        outx = outx.view(b, c, -1)
        outy = outy.view(b, c, -1)

    corr = cosine_similarity(outx, outy, dim=2).view(b, c)
    corr = torch.sigmoid(corr)
    corr = corr.view(b, c, 1, 1)

    return corr


class myGNN(MessagePassing):
    """From Visual Camera Re-Localization using Graph Neural Networks and Relative Pose Supervision."""
    def __init__(self, in_channels, out_channels, H, W, aggr="add", attention=False, pooling=None,
                 k=-1, first_GNN_layer=False, **kwargs):
        super(myGNN, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = H
        self.W = W
        self.attention = attention
        self.k = k
        self.pooling = pooling

        if first_GNN_layer:
            self.in_channels_edge = in_channels * 2 + 1
        else:
            self.in_channels_edge = in_channels * 2 + 1

        self.conv_message = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2+ out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_edge = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_edge, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_updating = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)

        # Define edge model
        self.edge_model = EdgeModel(self.conv_edge, self.in_channels, self.H, self.W,
                                    self.in_channels * 2)

        init_modules = [self.conv_message, self.conv_updating, self.conv_edge]

        self.init_parameters(init_modules=init_modules)

    def init_parameters(self, init_modules=None):
        if init_modules is None:
            return
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x, edge_index, edge_attr, batch=None, aggre_weights=None):

        # Edge Update
        if self.edge_model is not None:
            row, col = edge_index
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # Node Update
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, aggre_weights=aggre_weights)

        return out, edge_attr

    def message(self, x_i, x_j, edge_attr, aggre_weights=None):
        """
        :param x_i: E x in_channels
        :param x_j: E x in_channels
        :return: message
        """
        # *******update edge features here possible
        # self.__edgemat__ = edgemat

        x_i_full =  x_i.view(x_i.size(0), self.in_channels, self.H,
                            self.W).contiguous()  # recover the full size
        x_j_full =  x_j.view(x_j.size(0), self.in_channels, self.H,
                            self.W).contiguous()  # recover the full size
        edge_attr = edge_attr.view(edge_attr.size(0), -1, self.H,
                                   self.W).contiguous()  # recover the full size

        message = self.conv_message(torch.cat([x_i_full, x_j_full, edge_attr], dim=1).contiguous())
        #torch.Size([28680, 256, 1, 1])
        if aggre_weights is not None: message = aggre_weights[:, None, None, None] * message
        if self.attention:
            atten = batch_cosine_similarity(x_i_full, x_j_full, pooling=self.pooling)
            message = message * atten
        # num_pairs, output_channel
        return message.view(message.size(0), -1)

    def update(self, aggr_out, x):
        # two 3x3 layers for feature fusion
        x = x.view(x.size(0), self.in_channels, self.H,
                   self.W).contiguous()  # recover the full size

        aggr_out = aggr_out.view(aggr_out.size(0), self.out_channels, self.H,
                                 self.W).contiguous()  # recover the full size

        out = self.conv_updating(torch.cat([x, aggr_out], dim=1).contiguous())
        return out.view(out.size(0), -1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.

    This function solves the optimization problem and returns the O T matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Define the MLP module for edge classification
class EdgeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float=0.) -> None:
        """Init function for the EdgeMLP model.

        Args:
            input_dim: Dimension of the input features (concatenated embeddings)
            hidden_dim: Dimension of the hidden layer
            output_dim: Dimension of the output (number of classes)
        """
        super(EdgeMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        if dropout > 0: self.dropout_layer= nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the EdgeMLP model.

        Args:
            x: Input edge features (concatenated node embeddings)

        Returns:
            Output edge class logits
        """
        x = F.relu(self.fc1(x))
        if self.dropout > 0: x= self.dropout_layer(x)
        x = self.fc2(x)
        return x

class Custom_GNN_Predcitor(nn.Module):
    """"""
    def __init__(self,
            model_name='Custom_GNN_Predcitor',
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
            num_classes=1,
            dropout_mlp=0.,
            attention=False,
            return_edges=False,
            knn=-1
        ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.graph_transform = myGNN(8448, 256, 1, 1, attention=attention)
        self.edge_mlp = EdgeMLP(input_dim=token_dim, hidden_dim=256, output_dim=num_classes, dropout=dropout_mlp)
        self.return_edges = return_edges
        self.knn = knn

    def forward(self, nodes, edge_indices=None, aggre_weights=None):

        N = nodes.shape[0]
        if edge_indices is None:
            # similarity matrix S
            S = torch.einsum('id,jd->ij', nodes, nodes) # B, B
            diagonal = torch.eye(N, N).bool().to(S.device)
            S.masked_fill_(diagonal, -torch.inf)
            indices = list(range(N))
            # Generate all two-element combinations
            combinations = list(itertools.combinations(indices, 2))
            edge_indices = torch.tensor(combinations).T.to(S.device)

        else:
            S = torch.einsum('id,jd->ij', nodes, nodes) # B, B
            diagonal = torch.eye(N, N).bool().to(S.device)
            S.masked_fill_(diagonal, -torch.inf)

        edge_weights = S[edge_indices[0], edge_indices[1]]
        graph_data = Data(x=nodes, edge_index=edge_indices, edge_attr=edge_weights)
        nodes, updatd_edges = self.graph_transform(graph_data.x, graph_data.edge_index, graph_data.edge_attr, aggre_weights=aggre_weights)

        assert edge_indices.shape[1] == updatd_edges.shape[0]
        return_dic = {}
        return_dic['edges'] = self.edge_mlp(updatd_edges)
        return_dic['edge_indices'] = edge_indices
        return_dic['tokens'] = nodes
        if self.return_edges:
            return_dic['edge_features'] = updatd_edges

        return return_dic
