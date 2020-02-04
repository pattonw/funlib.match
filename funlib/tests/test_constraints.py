import networkx as nx
import numpy as np

import itertools
import pytest

import pylp

from funlib.match.get_constraints import get_constraints

from .gurobi_check import gurobi_installed_with_license
from .valid_matchings import (
    simple_chain,
    short_chain,
    long_chain,
    confounding_chain,
    confounding_loop,
    simple_4_branch,
)


@pytest.mark.parametrize(
    "data_func",
    [
        simple_chain,
        short_chain,
        long_chain,
        confounding_chain,
        confounding_loop,
        simple_4_branch,
    ],
)
@pytest.mark.parametrize(
    "use_gurobi", [pytest.param(True, marks=gurobi_installed_with_license()), False]
)
@pytest.mark.parametrize(
    "directed_overcomplete",
    [
        True,
        pytest.param(
            False, marks=pytest.mark.skip("undirected graphs not fully supported yet!")
        ),
    ],
)
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
        _,
        _,
        _,
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

    constraints = get_constraints(overcomplete, target, node_costs, edge_costs)

    for key, other_implementation in []:
        comparison_constraints = other_implementation(
            overcomplete, target, node_costs, edge_costs
        )

        for key in constraints.keys():
            cs, equiv, val = constraints[key]
            cs = set([tuple(sorted(x, key=lambda x: x[0])) for x in cs])
            other_cs, other_equiv, other_val = other_constraints[key]
            other_cs = set([tuple(sorted(x, key=lambda x: x[0])) for x in other_cs])

            assert val == other_val
            assert equiv == other_equiv
            assert cs == other_cs
