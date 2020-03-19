from funlib.match.helper_functions import match

import networkx as nx
import pytest

from .gurobi_check import gurobi_installed_with_license
from .valid_matchings import (
    simple_4_branch,
    simple_chain,
    short_chain,
    long_chain,
    confounding_chain,
    confounding_loop,
)


@pytest.mark.parametrize(
    "data_func",
    [
        simple_chain,
        short_chain,
        long_chain,
        simple_4_branch,
        confounding_chain,
        confounding_loop,
    ],
)
@pytest.mark.parametrize(
    "use_gurobi", [pytest.param(True, marks=gurobi_installed_with_license()), False]
)
@pytest.mark.parametrize("directed_overcomplete", [True, False])
def test_valid_matching(
    data_func, use_gurobi, directed_overcomplete, enforced_assignments=[]
):
    (
        target_nodes,
        target_edges,
        overcomplete_nodes,
        overcomplete_edges,
        node_costs,
        edge_costs,
        expected_node_matchings,
        expected_edge_matchings,
        expected_cost,
    ) = data_func()

    target = nx.DiGraph()
    target.add_nodes_from(target_nodes)
    target.add_edges_from(target_edges)

    if directed_overcomplete:
        overcomplete = nx.DiGraph()
    else:
        overcomplete = nx.Graph()
    overcomplete.add_nodes_from(overcomplete_nodes)
    overcomplete.add_edges_from(overcomplete_edges)

    matched = match(overcomplete, target, node_costs, edge_costs, use_gurobi=use_gurobi)

    for node, m in expected_node_matchings:
        if m is not None:
            assert node in matched.nodes()

    for edge, m in expected_edge_matchings:
        if m is None:
            assert edge not in matched.edges()
        else:
            assert edge in matched.edges()
