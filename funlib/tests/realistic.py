import pickle
from pathlib import Path


def large():
    """
    """

    example = pickle.load((Path(__file__).parent / "realistic_problem.obj").open("rb"))
    overcomplete = example["graph"]
    overcomplete_nodes = overcomplete.nodes
    overcomplete_edges = overcomplete.edges
    target = example["consensus"]
    target_nodes = target.nodes
    target_edges = target.edges
    node_costs = example["node_costs"]
    edge_costs = [((a, b), (c, d), e) for a, b, c, d, e in example["edge_costs"]]

    return (
        target_nodes,
        target_edges,
        overcomplete_nodes,
        overcomplete_edges,
        node_costs,
        edge_costs,
    )
