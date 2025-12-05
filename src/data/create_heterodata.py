import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import KMeans
import rasterio


class GraphData:
    def __init__(self, region, records_directory="data/raw/Records", *args, **kwargs):
        self.data = HeteroData()
        self.region = region
        self.records_directory = records_directory
        self.spacial_features, self.environmental_features = (
            self.get_location_and_environmental_features()
        )
        self.location_features = self.spacial_features + self.environmental_features

        # lets define the kwargs
        self.bg_data_method = kwargs.get("background_data_method", "from_csv")
        self.stratified_proportion = kwargs.get("stratified_proportion", 1.0)
        self.n_strata = kwargs.get("n_strata", 100)
        self.n_per_stratum = int((10000 * self.stratified_proportion) / self.n_strata)

        self.train_df = self.retrieve_po_data(region)
        self.train_df["location_id"] = self.train_df.groupby(
            self.location_features
        ).ngroup()

        self.bg_df = self.retrieve_background_data(method=self.bg_data_method)
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

    def retrieve_po_data(self, region):
        train_path = f"{self.records_directory}/train_po/{region}train_po.csv"
        train_df = pd.read_csv(train_path)
        return train_df

    def retrieve_background_data(self, method="from_csv"):
        if method == "from_csv":
            bg_path = f"{self.records_directory}/train_bg/{self.region}train_bg.csv"
            bg_df = pd.read_csv(bg_path)
            return bg_df[self.spacial_features + self.environmental_features]

        elif method == "preload_stratified_sampling":
            bg_path = f"data/interim/bg_stratified_sampled.csv"
            bg_df = pd.read_csv(bg_path)
            return bg_df[self.spacial_features + self.environmental_features]

        elif method == "stratified_sampling":
            bg_df = self.background_stratified_sampling()
            return bg_df[self.spacial_features + self.environmental_features]

    def background_stratified_sampling(self):
        src_dir = f"data/raw/Environment/{self.region}"
        first_env_feature = self.environmental_features[0]
        src_file_path = os.path.join(src_dir, f"{first_env_feature}.tif")
        # lets use this base raster file to get coordinates
        with rasterio.open(src_file_path) as base_src:
            height = base_src.height
            width = base_src.width
            n_pixels = height * width
            # get nsamples from any pixel
            n_samples = min(100000, n_pixels)
            samples_idx = np.random.choice(n_pixels, size=int(n_samples), replace=False)
            rows = samples_idx // width
            cols = samples_idx % width
            # get the coordinates in x, y from rows and cols
            xs, ys = rasterio.transform.xy(base_src.transform, rows, cols)

        # for the rest of environmental features, lets sample their values at the same locations
        sample_coordinates = list(zip(xs, ys))
        env_data = {}
        for feature in self.environmental_features:
            src_file_path = os.path.join(src_dir, f"{feature}.tif")
            with rasterio.open(src_file_path) as src:
                values = [x[0] for x in src.sample(sample_coordinates)]
                env_data[feature] = values

        bg_df = pd.DataFrame(env_data)
        bg_df["x"] = xs
        bg_df["y"] = ys

        # stratify by clustering the environmental features

        scaler = StandardScaler()
        env_features_scaled = scaler.fit_transform(
            bg_df[self.environmental_features].values
        )
        kmeans = KMeans(n_clusters=self.n_strata, random_state=42)
        strata_labels = kmeans.fit_predict(env_features_scaled)
        bg_df["strata"] = strata_labels

        # now sample from each strata n_per_stratum samples
        sampled_bg_dfs = []
        for stratum in range(self.n_strata):
            stratum_df = bg_df[bg_df["strata"] == stratum]
            if len(stratum_df) >= self.n_per_stratum:
                sampled_stratum_df = stratum_df.sample(
                    n=self.n_per_stratum, replace=False, random_state=42
                )
            else:
                sampled_stratum_df = stratum_df.sample(
                    n=self.n_per_stratum, replace=True, random_state=42
                )
            sampled_bg_dfs.append(sampled_stratum_df)
        bg_df = pd.concat(sampled_bg_dfs, ignore_index=True)
        # save the sampled background data for future use
        save_path = f"data/interim/bg_stratified_sampled.csv"
        bg_df.to_csv(save_path, index=False)
        return bg_df[self.spacial_features + self.environmental_features]

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
