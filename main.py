from src.data import create_heterodata
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

    graph_data.get_species_nodes(embedding_dim=16)
    graph_data.get_location_nodes(include_spacial_features=False)
    graph_data.get_edges(include_location_to_location_edges=True)

    hetero_data = graph_data.data

    check_heterodata_nulls(hetero_data)

    print(hetero_data)
