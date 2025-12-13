import os
import torch
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

    def retrieve_background_data(self, method="from_csv", save_clusters=False):
        if method == "from_csv":
            bg_path = f"{self.records_directory}/train_bg/{self.region}train_bg.csv"
            bg_df = pd.read_csv(bg_path)
            return bg_df[self.spacial_features + self.environmental_features]

        elif method == "preload_stratified_sampling":
            bg_path = f"data/processed/{self.region}_bg_cleaned.csv"
            bg_df = pd.read_csv(bg_path)
            return bg_df[self.spacial_features + self.environmental_features]

        elif method == "stratified_sampling":
            bg_df = self.background_stratified_sampling()
            if save_clusters:
                # save the sampled background data for future use
                save_path = f"data/interim/bg_stratified_sampled.csv"
                bg_df.to_csv(save_path, index=False)
            bg_df = self.clean_background_data(bg_df)

            save_path = f"data/processed/{self.region}_bg_cleaned.csv"
            if not os.path.exists("data/processed/"):
                os.makedirs("data/processed/")
            bg_df.to_csv(save_path, index=False)
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
        return bg_df

    def clean_background_data(self, bg_df, min_threshold=-1e5):
        bg_df_cleaned = bg_df[(bg_df >= min_threshold).all(axis=1)]
        return bg_df_cleaned

    def get_all_locations(self):
        po_locations = self.train_df[
            self.spacial_features + ["location_id"] + self.environmental_features
        ].drop_duplicates(subset=["location_id"])
        bg_locations = self.bg_df[
            self.spacial_features + ["location_id"] + self.environmental_features
        ].drop_duplicates(subset=["location_id"])
        return pd.concat([po_locations, bg_locations], ignore_index=True)

    def get_species_nodes(self):
        """
        Create species node features: [species_index, group_one_hot...]

        The model will use species_index to look up learned embeddings.
        Output shape: [num_unique_species, 1 + num_groups]
        """
        # Get unique species with their group
        species_df = (
            self.train_df[["spid", "group"]].drop_duplicates(subset=["spid"]).copy()
        )

        # Sort by species index to ensure alignment
        species_df["sp_idx"] = species_df["spid"].map(self.species_to_index)
        species_df = species_df.sort_values("sp_idx").reset_index(drop=True)

        # Verify alignment
        assert list(species_df["sp_idx"]) == list(
            range(self.num_species)
        ), "Species indices are not properly aligned!"

        # Species indices as first column
        species_indices = torch.tensor(
            species_df["sp_idx"].values, dtype=torch.float  # float for concatenation
        ).unsqueeze(
            1
        )  # Shape: [num_species, 1]

        # One-hot encode taxonomic group
        group_one_hot = pd.get_dummies(species_df["group"], prefix="group")
        group_features = torch.tensor(group_one_hot.values, dtype=torch.float)
        # Shape: [num_species, num_groups]

        # Combine: [species_idx, group_one_hot...]
        species_node_features = torch.cat([species_indices, group_features], dim=1)
        # Shape: [num_species, 1 + num_groups]

        self.data["species"].x = species_node_features
        self.species_df = species_df
        self.num_groups = group_one_hot.shape[1]

        print(f"Species nodes: {species_node_features.shape}")
        print(f"  - Format: [species_idx, group_one_hot...]")
        print(f"  - Num species: {self.num_species}")
        print(f"  - Num groups: {self.num_groups}")

    def get_location_nodes(self, include_spacial_features=False, normalize=True):
        """
        Create location node features from environmental variables.

        Output shape: [num_locations, num_env_features]
        """
        if include_spacial_features:
            feature_columns = self.spacial_features + self.environmental_features
        else:
            feature_columns = self.environmental_features

        location_features = self.all_locations_df[feature_columns].values

        if normalize:
            self.location_scaler = StandardScaler()
            location_features = self.location_scaler.fit_transform(location_features)

        self.data["location"].x = torch.tensor(location_features, dtype=torch.float)
        print(f"Location nodes created: {self.data['location'].x.shape}")
        print(f"  - Features: {feature_columns}")
        print(f"  - Normalized: {normalize}")

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
                np.array([row_indices, col_indices]), dtype=torch.long
            )

    def get_training_edges(self, negative_ratio=1, neg_sampling_strategy="balanced"):
        """
        Generate positive and negative edges for training.

        Parameters:
        -----------
        negative_ratio : float
            Ratio of negative to positive examples (1.0 = equal, 2.0 = twice as many negatives)
        neg_sampling_strategy : str
            "balanced": equal negatives from PO and BG locations
            "po_only": negatives only from PO locations
            "bg_only": negatives only from background locations
            "proportional": proportional to number of locations of each type

        Returns:
        --------
        pos_edge_index : torch.Tensor [2, num_pos]
        neg_edge_index : torch.Tensor [2, num_neg]
        """
        # Positive edges
        species_indices = [self.species_to_index[sid] for sid in self.train_df["spid"]]
        location_indices = [
            self.location_to_idx[loc_id] for loc_id in self.train_df["location_id"]
        ]
        positive_edge_index = torch.tensor(
            [species_indices, location_indices], dtype=torch.long
        )
        num_positive = positive_edge_index.shape[1]

        # Negative edges
        # Identify PO and BG location indices
        po_location_ids = self.train_df["location_id"].unique()
        bg_location_ids = self.bg_df["location_id"].unique()

        # Now we are going to get the number of species present in each location
        location_to_observed_species = {}
        for _, row in self.train_df.iterrows():
            loc_id = row["location_id"]
            spid = row["spid"]
            if loc_id not in location_to_observed_species:
                location_to_observed_species[loc_id] = set()
            location_to_observed_species[loc_id].add(spid)

        all_species_ids = set(self.species_to_index.keys())
        # Now we create the negatives from PO locations
        neg_edges_po = []
        for loc_id in po_location_ids:
            loc_idx = self.location_to_idx[loc_id]
            observed_species = location_to_observed_species[loc_id]
            unobserved_species = all_species_ids - observed_species
            for spid in unobserved_species:
                sp_idx = self.species_to_index[spid]
                neg_edges_po.append((sp_idx, loc_idx))

        # Now create the negatives from BG locations
        neg_edges_bg = []
        for loc_id in self.bg_location_ids:
            loc_idx = self.location_to_idx[loc_id]
            for sp_idx in all_species_ids:
                neg_edges_bg.append((sp_idx, loc_idx))

        num_negatives = int(num_positive * negative_ratio)

        if neg_sampling_strategy == "balanced":
            num_neg_po = num_negatives // 2
            num_neg_bg = num_negatives - num_neg_po
            sampled_po = self._sample_from_list(neg_edges_po, num_neg_po)
            sampled_bg = self._sample_from_list(neg_edges_bg, num_neg_bg)
            negative_paris = sampled_po + sampled_bg

        elif neg_sampling_strategy == "po_only":
            negative_pairs = self._sample_from_list(neg_edges_po, num_negatives)

        elif neg_sampling_strategy == "bg_only":
            negative_pairs = self._sample_from_list(neg_edges_bg, num_negatives)

        elif neg_sampling_strategy == "proportional":
            total = len(neg_edges_po) + len(neg_edges_bg)
            po_fraction = len(neg_edges_po) / total if total > 0 else 0.5
            n_from_po = int(num_negatives * po_fraction)
            n_from_bg = num_negatives - n_from_po
            sampled_po = self._sample_from_list(neg_edges_po, n_from_po)
            sampled_bg = self._sample_from_list(neg_edges_bg, n_from_bg)
            negative_pairs = sampled_po + sampled_bg
        else:
            raise ValueError(f"Unknown strategy: {neg_sampling_strategy}")

        # Convert to tensor
        if len(negative_pairs) > 0:
            neg_species = [pair[0] for pair in negative_pairs]
            neg_locations = [pair[1] for pair in negative_pairs]
            negative_edge_index = torch.tensor(
                [neg_species, neg_locations], dtype=torch.long
            )
        else:
            negative_edge_index = torch.tensor([[], []], dtype=torch.long)

        print(f"Training edges generated:")
        print(f"  - Positives: {num_positive}")
        print(
            f"  - Negatives: {negative_edge_index.shape[1]} ({neg_sampling_strategy})"
        )
        print(f"  - Ratio: {negative_edge_index.shape[1] / num_positive:.2f}")

        return positive_edge_index, negative_edge_index

    def _sample_from_list(self, candidates, n_samples):
        """Sample n items from a list without replacement."""
        if len(candidates) == 0:
            return []
        if len(candidates) <= n_samples:
            return candidates
        indices = np.random.choice(len(candidates), size=n_samples, replace=False)
        return [candidates[i] for i in indices]

    def get_training_data(
        self, negative_ratio=1.0, neg_sampling_strategy="balanced", shuffle=True
    ):
        """
        Get combined training edges and labels.

        Returns
        -------
        edge_index : torch.Tensor [2, num_edges]
        labels : torch.Tensor [num_edges]
        """
        pos_edge_index, neg_edge_index = self.get_training_edges(
            negative_ratio=negative_ratio, neg_sampling_strategy=neg_sampling_strategy
        )

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat(
            [torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]
        )

        if shuffle:
            perm = torch.randperm(edge_index.shape[1])
            edge_index = edge_index[:, perm]
            labels = labels[perm]

        return edge_index, labels
