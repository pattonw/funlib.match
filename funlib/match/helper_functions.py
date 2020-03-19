import networkx as nx

from typing import Hashable, Tuple, List
import itertools
import copy

from .graph_to_tree_matcher import GraphToTreeMatcher, Node, Edge, Matchable


def build_matched(
    graph: nx.Graph,
    node_matchings: List[Tuple[Node, Node]],
    edge_matchings: List[Tuple[Edge, Edge]],
    target_attr:str = "target",
):
    """
    Given a graph, and a set of node and edge matchings, this function
    returns a subgraph of G after removing nodes and edges that weren't matched
    """
    edge_matchings = [
        (graph_e, tree_e) for graph_e, tree_e in edge_matchings if tree_e is not None
    ]
    nodes = set(itertools.chain(*[graph_e for graph_e, tree_e in edge_matchings]))
    node_matchings = [
        (graph_n, tree_n) for graph_n, tree_n in node_matchings if graph_n in nodes
    ]

    matched = type(graph)()
    matched.add_nodes_from(
        [
            (graph_n, {target_attr: tree_n, **copy.deepcopy(graph.nodes[graph_n])})
            for graph_n, tree_n in node_matchings
        ]
    )
    matched.add_edges_from(
        [
            (u, v, {target_attr: tree_e, **copy.deepcopy(graph.edges[(u, v)])})
            for (u, v), tree_e in edge_matchings
        ]
    )

    return matched


def match(
    graph: nx.Graph,
    tree: nx.DiGraph,
    node_match_costs: List[Tuple[Node, Node, float]],
    edge_match_costs: List[Tuple[Edge, Edge, float]],
    use_gurobi=True,
) -> nx.Graph:
    node_matchings, edge_matchings, _ = GraphToTreeMatcher(
        graph, tree, node_match_costs, edge_match_costs, use_gurobi=use_gurobi
    ).match()
    matched = build_matched(graph, node_matchings, edge_matchings)
    return matched

