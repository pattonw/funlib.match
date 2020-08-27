import networkx as nx
import numpy as np

import itertools
import pytest

from funlib.match import GraphToTreeMatcher

from .valid_matchings import (
    simple_chain,
    short_chain,
    long_chain,
    confounding_chain,
    confounding_loop,
    simple_4_branch,
)

from .invalid_matchings import not_overcomplete

from .realistic import large

from .gurobi_check import gurobi_installed_with_license


@pytest.mark.parametrize(
    "use_gurobi", [pytest.param(True, marks=gurobi_installed_with_license()), False]
)
@pytest.mark.parametrize("directed_overcomplete", [True, False])
def test_enforced_assignment(use_gurobi, directed_overcomplete):
    """
    target graph:

       A---->B---->C

    overcomplete graph:

    a--b--c--d--e--f--g

    Force a->A, c->C, and e->C
    """

    expected_node_matchings = [
        ("a", "A"),
        ("b", None),
        ("c", "B"),
        ("d", None),
        ("e", "C"),
        ("f", None),
        ("g", None),
    ]

    expected_edge_matchings = [
        (("a", "b"), ("A", "B")),
        (("b", "c"), ("A", "B")),
        (("c", "d"), ("B", "C")),
        (("d", "e"), ("B", "C")),
    ]

    expected_cost = 27

    enforced_assignments = list(
        itertools.chain(expected_node_matchings, expected_edge_matchings[1:])
    )

    def enforced_long_chain():
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
        ) = long_chain()
        return (
            target_nodes,
            target_edges,
            overcomplete_nodes,
            overcomplete_edges,
            node_costs,
            edge_costs,
            expected_node_matchings,
            expected_edge_matchings,
            expected_cost,
        )

    test_valid_matching(
        enforced_long_chain,
        use_gurobi,
        directed_overcomplete,
        enforced_assignments=enforced_assignments,
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

    matcher = GraphToTreeMatcher(
        overcomplete, target, node_costs, edge_costs, use_gurobi=use_gurobi
    )
    matcher.enforce_expected_assignments(enforced_assignments)
    node_matchings, edge_matchings, cost = matcher.match()
    matchings = {a: b for a, b in itertools.chain(node_matchings, edge_matchings)}

    expected_matchings = tuple(
        [x for x in itertools.chain(expected_node_matchings, expected_edge_matchings)]
    )
    seen_matchings = tuple(
        [
            (a, matchings[a])
            for a, _ in itertools.chain(
                expected_node_matchings, expected_edge_matchings
            )
        ]
    )
    assert expected_matchings == seen_matchings

    assert cost == expected_cost


@pytest.mark.parametrize(
    "data_func", [not_overcomplete,],
)
@pytest.mark.parametrize(
    "use_gurobi", [pytest.param(True, marks=gurobi_installed_with_license()), False]
)
@pytest.mark.parametrize("directed_overcomplete", [True, False])
def test_invalid_matching(data_func, use_gurobi, directed_overcomplete):
    (
        target_nodes,
        target_edges,
        overcomplete_nodes,
        overcomplete_edges,
        node_costs,
        edge_costs,
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

    with pytest.raises(ValueError, match=r"Optimal solution \*NOT\* found"):
        node_matchings, edge_matchings, cost = GraphToTreeMatcher(
            overcomplete, target, node_costs, edge_costs, use_gurobi=use_gurobi
        ).match()

