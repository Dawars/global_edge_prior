import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP (concat(e1, e2))
    """
    def __init__(
            self,
            model_name='mlp',
            num_channels=8448,
            out_channels=256, 
            edge_filtered=False,
            dropout_mlp=0.,
            dustbin=False,
            sigmoid_shift=False, return_tokens = False
        ):
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.edge_filtered = edge_filtered
        self.dustbin = dustbin
        self.sigmoid_shift = sigmoid_shift

        self.fc1 = nn.Linear(num_channels*2, out_channels)
        self.fc2 = nn.Linear(out_channels, 1)
        self.dropout = dropout_mlp
        if dropout_mlp > 0: self.dropout_layer= nn.Dropout(dropout_mlp)
        self.return_tokens = return_tokens
        # if dropout_mlp > 0:
        #     self.edge_classifier = nn.Sequential(
        #     nn.Linear(num_channels*2, out_channels),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_mlp),
        #     nn.Linear(out_channels, 1),
        # )
        # else:
        #     self.edge_classifier = nn.Sequential(
        #         nn.Linear(num_channels*2, out_channels),
        #         nn.ReLU(),
        #         nn.Linear(out_channels, 1),
        #     )
        if sigmoid_shift: self.sigmoid_shift_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, nodes, edge_indices=None):

        N = nodes.shape[0]
        if self.edge_filtered:
            if edge_indices is None: 
                # similarity matrix S
                S = torch.einsum('id,jd->ij', nodes, nodes) # B, B
                diagonal = torch.eye(N, N).bool().to(S.device)
                S.masked_fill_(diagonal, -torch.inf)
                threshold = S.median().item()
                edge_indices = (S > threshold).nonzero(as_tuple=False).T  # (2, num_edges)
        else:
            indices = list(range(N))
            # Generate all two-element combinations
            combinations = list(itertools.combinations(indices, 2))
            edge_indices = torch.tensor(combinations).T
        # return edge class logits, and indices

        x = F.relu(self.fc1(torch.concat((nodes[edge_indices[0]], nodes[edge_indices[1]]), dim=1)))
        if self.dropout > 0: x= self.dropout_layer(x)
        x = self.fc2(x)
        return {
            'edges': x,#self.edge_classifier(torch.concat((nodes[edge_indices[0]], nodes[edge_indices[1]]), dim=1)),
            'edge_indices': edge_indices,
            'tokens': nodes
        } if self.return_tokens else {
            'edges': x ,#self.edge_classifier(torch.concat((nodes[edge_indices[0]], nodes[edge_indices[1]]), dim=1)),
            'edge_indices': edge_indices,
        }


# class EdgeMLP(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float=0.) -> None:
#         """Init function for the EdgeMLP model.

#         Args:
#             input_dim: Di
#             mension of the input features (concatenated embeddings)
#             hidden_dim: Dimension of the hidden layer
#             output_dim: Dimension of the output (number of classes)
#         """
#         super(EdgeMLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.dropout = dropout
#         if dropout > 0: self.dropout_layer= nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward function for the EdgeMLP model.

#         Args:
#             x: Input edge features (concatenated node embeddings)

#         Returns:
#             Output edge class logits
#         """
#         x = F.relu(self.fc1(x))
#         if self.dropout > 0: x= self.dropout_layer(x)
#         x = self.fc2(x)
#         return x   