from src.data import create_heterodata


if __name__ == "__main__":
    graph_data = create_heterodata.GraphData(
        region="AWT",
        background_data_method="stratified_sampling",
        n_background_samples=50000,
        stratified_proportion=1.0,
        n_strata=100,
    )

    graph_data.get_species_nodes(embedding_dim=16)
    graph_data.get_location_nodes(include_spacial_features=False)
    graph_data.get_edges(include_location_to_location_edges=True)

    hetero_data = graph_data.data

    print(hetero_data)
