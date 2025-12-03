import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import HeteroData


class GraphData:
    def __init__(self, region, records_directory="data/raw/Records"):
        self.data = HeteroData()
        self.region = region
        self.records_directory = records_directory
        self.train_df, self.bg_df = self.retrieve_region_data(region)
        
    def retrieve_region_data(self, region):
        train_path = f"{self.records_directory}/train_po/{region}train_po.csv"
        bg_path = f"{self.records_directory}/train_bg/{region}train_bg.csv"
        train_df = pd.read_csv(train_path)
        bg_df = pd.read_csv(bg_path)
        return train_df, bg_df

    def get_species_nodes():
        pass  # Implementation would go here

    def get_location_nodes():
        pass  # Implementation would go here

    def get_edges():
        pass  # Implementation would go here
        



def get_node_species_features(df, embedding_dim=16):
    # one hot encode the taxonomic group feature
    group_one_hot = pd.get_dummies(df["group"], prefix="group")
    group_features = torch.tensor(group_one_hot.values, dtype=torch.float)
    # we need to create a mapping from species id to an integer index
    unique_species = df["spid"].unique()
    species_id_to_index = {
        species_id: index for index, species_id in enumerate(unique_species)
    }
    num_species = len(unique_species)
    # tensor with the species_id indices
    species_indices = torch.tensor(
        [species_id_to_index[sid] for sid in df["spid"]], dtype=torch.long
    )
    species_embedding = nn.Embedding(num_species, embedding_dim)
    species_embedded = species_embedding(species_indices)
    # Now we can create the node features for species nodes by concatenating the embedded species_id with the one hot encoded group features
    species_node_features = torch.cat([species_embedded, group_features], dim=1)
    return species_node_features


def create_heterodata(df):
    # first we create a HereteroData object
    data = HeteroData()
    target_label = ["occ"]
    location_features = [
        "bc01",
        "bc04",
        "bc05",
        "bc06",
        "bc12",
        "bc15",
        "bc17",
        "bc20",
        "bc31",
        "bc33",
        "slope",
        "topo",
        "tri",
        "x",
        "y",
    ]
    species_features = ["group"]
    data["species"].x = get_node_species_features(df, embedding_dim=16)
    data["location"].x = torch.tensor(df[location_features].values, dtype=torch.float)
    data["location"].y = torch.tensor(df[target_label].values, dtype=torch.float)

    # Now we create the edges
    unique_species = df["spid"].unique()
    species_id_to_index = {
        species_id: index for index, species_id in enumerate(unique_species)
    }
    unique_locations = df["siteid"].unique()
    location_id_to_index = {
        location_id: index for index, location_id in enumerate(unique_locations)
    }

    mappings = {
        "species_to_idx": species_id_to_index,
        "location_to_idx": location_id_to_index,
        "idx_to_species": {v: k for k, v in species_id_to_index.items()},
        "idx_to_location": {v: k for k, v in location_id_to_index.items()},
    }
    species_indices = [species_id_to_index[sid] for sid in df["spid"]]
    location_indices = [location_id_to_index[loc_id] for loc_id in df["siteid"]]

    data["species", "observed_at", "location"].edge_index = torch.tensor(
        [species_indices, location_indices], dtype=torch.long
    )
    data["location", "observes", "species"].edge_index = torch.tensor(
        [location_indices, species_indices], dtype=torch.long
    )

    return data, mappings
