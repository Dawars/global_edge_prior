import torch
import torch.nn as nn
import itertools

class WeightedDotProductClassifier(nn.Module):
    def __init__(self, name='dotproduct', embedding_dim=8448, output_dim=256, return_tokens=False, aggre_weights=None):
        super(WeightedDotProductClassifier, self).__init__()
        self.return_tokens = return_tokens

    def forward(self, nodes, edge_indices=None):
        N = nodes.shape[0]
        if edge_indices == None: 
            S = torch.einsum('id,jd->ij', nodes, nodes) # B, B
            diagonal = torch.eye(N, N).bool().to(S.device)
            S.masked_fill_(diagonal, -torch.inf)
            indices = list(range(N))
            # Generate all two-element combinations
            combinations = list(itertools.combinations(indices, 2))
            edge_indices = torch.tensor(combinations).T.to(S.device)
        edge_weights = torch.einsum('ij,ij->i', nodes[edge_indices[0]], nodes[edge_indices[1]])

        # print(self.W)
        return {
            'edges': edge_weights,
            'edge_indices': edge_indices,
            'tokens': nodes
    } if self.return_tokens else{
            'edges': edge_weights,
            'edge_indices': edge_indices,
        }