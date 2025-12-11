import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv
import torch.nn.functional as F


class EcologicalHeteroGNN(nn.Module):
    def __init__(
        self, species_input_dim, location_input_dim, hidden_dim, out_dim, heads
    ):
        super().__init__()

        # first thing to do is project the input nodes to same dimension
        self.species_lin = nn.Linear(species_input_dim, hidden_dim)
        self.location_lin = nn.Linear(location_input_dim, hidden_dim)

        # Create HeteroConv with Graph Attention layers for each edge type
        self.conv1 = HeteroConv(
            {
                # First the species node embeddings from location nodes
                ("location", "observes", "species"): GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=0.2,
                    add_self_loops=False,
                ),
                # Second the location node embeddings, first from species messages
                ("species", "observed_at", "location"): GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=0.2,
                    add_self_loops=False,
                ),
                # Then the location embeddings from other location messages
                ("location", "nearby", "location"): GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=0.2,
                    add_self_loops=True,
                ),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("location", "observes", "species"): GATConv(
                    hidden_dim,
                    out_dim // heads,
                    heads=heads,
                    dropout=0.2,
                    add_self_loops=False,
                ),
                ("species", "observed_at", "location"): GATConv(
                    hidden_dim,
                    out_dim // heads,
                    heads=heads,
                    dropout=0.2,
                    add_self_loops=False,
                ),
                ("location", "nearby", "location"): GATConv(
                    hidden_dim,
                    out_dim // heads,
                    heads=heads,
                    dropout=0.2,
                    add_self_loops=True,
                ),
            },
            aggr="sum",
        )

    def forward(self, data):
        # Initial projection to common dimension
        h = {
            "species": self.species_lin(data["species"].x),
            "location": self.location_lin(data["location"].x),
        }

        # Layer 1 with skip connection + activation
        h_new = self.conv1(h, data.edge_index_dict)
        h = {key: F.relu(h_new[key] + h[key]) for key in h}

        # Layer 2 with skip connection (no activation - final layer)
        h_new = self.conv2(h, data.edge_index_dict)
        h = {key: h_new[key] + h[key] for key in h}

        return h["species"], h["location"]

    def decode(self, h_species, h_location, edge_label_index):
        src, dst = edge_label_index
        edge_embeddings = h_species[src] * h_location[dst]
        return torch.sum(edge_embeddings, dim=-1)
