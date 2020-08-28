import pickle
from pathlib import Path

import networkx as nx

from funlib.match import GraphToTreeMatcher


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

def test_large():
    (
        target_nodes,
        target_edges,
        overcomplete_nodes,
        overcomplete_edges,
        node_costs,
        edge_costs,
    ) = large()
    target = nx.DiGraph()
    target.add_nodes_from(target_nodes)
    target.add_edges_from(target_edges)

    overcomplete = nx.Graph()
    overcomplete.add_nodes_from(overcomplete_nodes)
    overcomplete.add_edges_from(overcomplete_edges)

    matcher = GraphToTreeMatcher(
        overcomplete, target, node_costs, edge_costs, use_gurobi=False
    )
    node_matchings, edge_matchings, cost = matcher.match()