import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.neighbors import radius_neighbors_graph


class GraphData:
    def __init__(self, region, records_directory="data/raw/Records"):
        self.data = HeteroData()
        self.region = region
        self.records_directory = records_directory
        self.train_df, self.bg_df = self.retrieve_region_data(region)
        self.spacial_features, self.environmental_features = (
            self.get_location_and_environmental_features()
        )
        self.location_features = self.spacial_features + self.environmental_features
        self.train_df["location_id"] = self.train_df.groupby(
            self.location_features
        ).ngroup()
        self.bg_df["location_id"] = (
            self.bg_df.groupby(self.location_features).ngroup()
            + self.train_df["location_id"].nunique()
        )
        self.all_locations_df = self.get_all_locations()
        self.location_to_idx = {
            loc_id: idx
            for idx, loc_id in enumerate(self.all_locations_df["location_id"])
        }
        self.species_to_index = {
            species_id: index
            for index, species_id in enumerate(self.train_df["spid"].unique())
        }
        self.num_locations = len(self.all_locations_df)
        self.num_species = len(self.species_to_index)

    def retrieve_region_data(self, region):
        train_path = f"{self.records_directory}/train_po/{region}train_po.csv"
        bg_path = f"{self.records_directory}/train_bg/{region}train_bg.csv"
        train_df = pd.read_csv(train_path)
        bg_df = pd.read_csv(bg_path)
        return train_df, bg_df

    def get_location_and_environmental_features(self):
        if self.region == "AWT":
            spacial_features = ["x", "y"]
            environmental_features = [
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
            ]
        return spacial_features, environmental_features

    def get_all_locations(self):
        po_locations = self.train_df[
            self.spacial_features + ["location_id"] + self.environmental_features
        ].drop_duplicates(subset=["location_id"])
        bg_locations = self.bg_df[
            self.spacial_features + ["location_id"] + self.environmental_features
        ].drop_duplicates(subset=["location_id"])
        return pd.concat([po_locations, bg_locations], ignore_index=True)

    def get_species_nodes(self, embedding_dim=16):
        group_one_hot = pd.get_dummies(self.train_df["group"], prefix="group")
        group_features = torch.tensor(group_one_hot.values, dtype=torch.float)
        species_embedding = nn.Embedding(self.num_species, embedding_dim)
        species_indices = torch.tensor(
            [self.species_to_index[sid] for sid in self.train_df["spid"]],
            dtype=torch.long,
        )
        species_embedded = species_embedding(species_indices)
        species_node_features = torch.cat([species_embedded, group_features], dim=1)
        self.data["species"].x = species_node_features

    def get_location_nodes(self, include_spacial_features=False):
        if include_spacial_features:
            location_features = self.all_locations_df[
                self.spacial_features + self.environmental_features
            ]
        else:
            location_features = self.all_locations_df[self.environmental_features]

        self.data["location"].x = torch.tensor(
            location_features.values, dtype=torch.float
        )

    def get_edges(self, include_location_to_location_edges=False, radius=7500):
        species_indices = [self.species_to_index[sid] for sid in self.train_df["spid"]]
        location_indices = [
            self.location_to_idx[loc_id] for loc_id in self.train_df["location_id"]
        ]
        self.data["species", "observed_at", "location"].edge_index = torch.tensor(
            [species_indices, location_indices], dtype=torch.long
        )
        self.data["location", "observes", "species"].edge_index = torch.tensor(
            [location_indices, species_indices], dtype=torch.long
        )
        if include_location_to_location_edges:
            spatial_graph = radius_neighbors_graph(
                self.all_locations_df[self.spacial_features].values,
                radius=radius,
                mode="connectivity",
            )
            row_indices, col_indices = spatial_graph.nonzero()
            self.data["location", "nearby", "location"].edge_index = torch.tensor(
                [row_indices, col_indices], dtype=torch.long
            )
