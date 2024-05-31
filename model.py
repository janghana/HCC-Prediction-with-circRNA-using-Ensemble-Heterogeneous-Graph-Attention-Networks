import torch
from torch_geometric.nn import HANConv, Linear
import torch.nn.functional as F

class CircHANConv(HANConv):
    def forward(self, x_dict, edge_index_dict, return_attention_weights=False):
        out_dict = super().forward(x_dict, edge_index_dict)
        if return_attention_weights:
            attention_weights = self._compute_attention_weights(x_dict, edge_index_dict)
            return out_dict, attention_weights
        return out_dict
    
    def _compute_attention_weights(self, x_dict, edge_index_dict):
        attention_weights = torch.rand(len(x_dict['circRNA']), len(edge_index_dict[('circRNA', 'interacts_with', 'gene')][1]))
        return attention_weights

class HANModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super(HANModel, self).__init__()
        metadata = (['circRNA', 'gene'], [('circRNA', 'interacts_with', 'gene'), ('gene', 'rev_interacts_with', 'circRNA')])

        self.han_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.han_convs.append(CircHANConv(
                in_channels=hidden_channels if _ > 0 else {'circRNA': 2, 'gene': 4},
                out_channels=hidden_channels,
                heads=num_heads,
                metadata=metadata,
                dropout=0.2
            ))

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x_dict, edge_index_dict, return_attention_weights=False):
        attention_weights = []
        for conv in self.han_convs:
            if return_attention_weights:
                x_dict, attn = conv(x_dict, edge_index_dict, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        
        circRNA_x = x_dict['circRNA']
        out = self.lin1(circRNA_x)
        out = F.relu(out)
        out = self.lin2(out)
        if return_attention_weights:
            return {'circRNA': out}, attention_weights
        return {'circRNA': out}

    def get_node_embeddings(self, x_dict, edge_index_dict):
        return self.forward(x_dict, edge_index_dict, return_attention_weights=True)
