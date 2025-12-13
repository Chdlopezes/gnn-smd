from src.data import create_heterodata
from src.models.heterognn import EcologicalHeteroGNN
import torch


def check_heterodata_nulls(hetero_data):
    for node_type in hetero_data.node_types:
        node_data = hetero_data[node_type]
        if torch.isnan(node_data.x).any():
            print(f"Node type '{node_type}' has NaN values in its features.")
        else:
            print(f"Node type '{node_type}' has no NaN values in its features.")

    for edge_type in hetero_data.edge_types:
        edge_data = hetero_data[edge_type]
        if edge_data.edge_index is None:
            print(f"Edge type '{edge_type}' has no edges defined.")
        else:
            print(f"Edge type '{edge_type}' has edges defined.")


if __name__ == "__main__":
    graph_data = create_heterodata.GraphData(
        region="AWT",
        background_data_method="preload_stratified_sampling",
        stratified_proportion=1.0,
        n_strata=100,
    )

    graph_data.get_species_nodes()
    graph_data.get_location_nodes(include_spacial_features=False, normalize=True)
    graph_data.get_edges(include_location_to_location_edges=True)

    hetero_data = graph_data.data

    print(hetero_data["species"].x.shape)  # [40, 3] if 2 groups: [idx, group1, group2]
    print(hetero_data["species"].x[:5])

    # Create model
    model = EcologicalHeteroGNN(
        num_species=graph_data.num_species,
        num_groups=graph_data.num_groups,
        location_input_dim=len(graph_data.environmental_features),
        species_embedding_dim=16,
        hidden_dim=32,
        heads=4,
    )

    h_species, h_location = model(hetero_data)
    print(h_species.shape)  # [40, 32]
    print(h_location.shape)  # [10638, 32]
    print(model.species_embedding.weight.shape)
