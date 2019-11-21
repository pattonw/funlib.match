import networkx as nx
import numpy as np

import itertools
import pytest

from funlib.match import GraphToTreeMatcher


def validate_matching(
    target_nodes,
    target_edges,
    overcomplete_nodes,
    overcomplete_edges,
    node_costs,
    edge_costs,
    expected_node_matchings,
    expected_edge_matchings,
    expected_cost,
    enforced_assignments=[],
):
    target = nx.DiGraph()
    target.add_nodes_from(target_nodes)
    target.add_edges_from(target_edges)

    overcomplete = nx.Graph()
    overcomplete.add_nodes_from(overcomplete_nodes)
    overcomplete.add_edges_from(overcomplete_edges)

    matcher = GraphToTreeMatcher(overcomplete, target, node_costs, edge_costs)
    matcher.enforce_expected_assignments(enforced_assignments)
    node_matchings, edge_matchings, cost = matcher.match()
    matchings = {a: b for a, b in itertools.chain(node_matchings, edge_matchings)}

    matcher2 = GraphToTreeMatcher(
        overcomplete, target, node_costs, edge_costs, use_gurobi=False
    )
    matcher2.enforce_expected_assignments(enforced_assignments)
    node_matchings2, edge_matchings2, cost2 = matcher2.match()
    matchings2 = {a: b for a, b in itertools.chain(node_matchings2, edge_matchings2)}

    for a, b in itertools.chain(expected_node_matchings, expected_edge_matchings):
        assert matchings[a] == matchings2[a] == b

    assert cost == cost2 == expected_cost

    overcomplete = nx.DiGraph()
    overcomplete.add_nodes_from(overcomplete_nodes)
    overcomplete.add_edges_from(overcomplete_edges)

    matcher = GraphToTreeMatcher(overcomplete, target, node_costs, edge_costs)
    matcher.enforce_expected_assignments(enforced_assignments)
    node_matchings, edge_matchings, cost = matcher.match()
    matchings = {a: b for a, b in itertools.chain(node_matchings, edge_matchings)}

    matcher2 = GraphToTreeMatcher(
        overcomplete, target, node_costs, edge_costs, use_gurobi=False
    )
    matcher2.enforce_expected_assignments(enforced_assignments)
    node_matchings2, edge_matchings2, cost2 = matcher2.match()
    matchings2 = {a: b for a, b in itertools.chain(node_matchings2, edge_matchings2)}

    for a, b in itertools.chain(expected_node_matchings, expected_edge_matchings):
        assert matchings[a] == matchings2[a] == b

    assert cost == cost2 == expected_cost


def test_simple_chain():
    """
    target graph:

    A---->B---->C

    overcomplete graph:

    a--b--c--d--e
    """
    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b", "c", "d", "e"]
    overcomplete_edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]

    node_costs = [
        ("a", "A", 1),
        ("b", "A", 5),
        ("b", "B", 5),
        ("c", "B", 1),
        ("d", "B", 5),
        ("d", "C", 5),
        ("e", "C", 1),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("b", "c"), ("A", "B"), 1),
        (("b", "c"), ("B", "C"), 5),
        (("c", "d"), ("A", "B"), 5),
        (("c", "d"), ("B", "C"), 1),
        (("d", "e"), ("B", "C"), 1),
    ]

    expected_node_matchings = [
        ("a", "A"),
        ("b", None),
        ("c", "B"),
        ("d", None),
        ("e", "C"),
    ]

    expected_edge_matchings = [
        (("a", "b"), ("A", "B")),
        (("b", "c"), ("A", "B")),
        (("c", "d"), ("B", "C")),
        (("d", "e"), ("B", "C")),
    ]

    expected_cost = 7

    validate_matching(
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


def test_short_chain():
    """
    target graph:

    A---->B---->C

    overcomplete graph:

        a--b--c
    """
    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b", "c"]
    overcomplete_edges = [("a", "b"), ("b", "c")]

    node_costs = [
        ("a", "A", 5),
        ("a", "B", 5),
        ("b", "B", 1),
        ("c", "B", 5),
        ("c", "C", 5),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("a", "b"), ("B", "C"), 5),
        (("b", "c"), ("A", "B"), 5),
        (("b", "c"), ("B", "C"), 1),
    ]

    expected_node_matchings = [
        ("a", "A"),
        ("b", "B"),
        ("c", "C"),
    ]

    expected_edge_matchings = [
        (("a", "b"), ("A", "B")),
        (("b", "c"), ("B", "C")),
    ]

    expected_cost = 13

    validate_matching(
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


def test_long_chain():
    """
    target graph:

        A---->B---->C

    overcomplete graph:

    a--b--c--d--e--f--g

    matching should not have too many edge assignments.
    """

    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b", "c", "d", "e", "f", "g"]
    overcomplete_edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
    ]

    node_costs = [
        ("a", "A", 5),
        ("b", "A", 1),
        ("c", "A", 5),
        ("c", "B", 5),
        ("d", "B", 1),
        ("e", "B", 5),
        ("e", "C", 5),
        ("f", "C", 1),
        ("g", "A", 5),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 5),
        (("b", "c"), ("A", "B"), 1),
        (("c", "d"), ("A", "B"), 1),
        (("c", "d"), ("B", "C"), 5),
        (("d", "e"), ("A", "B"), 5),
        (("d", "e"), ("B", "C"), 1),
        (("e", "f"), ("B", "C"), 1),
        (("f", "g"), ("B", "C"), 5),
    ]

    expected_node_matchings = [
        ("a", None),
        ("b", "A"),
        ("c", None),
        ("d", "B"),
        ("e", None),
        ("f", "C"),
        ("g", None),
    ]

    expected_edge_matchings = [
        (("a", "b"), None),
        (("b", "c"), ("A", "B")),
        (("c", "d"), ("A", "B")),
        (("d", "e"), ("B", "C")),
        (("e", "f"), ("B", "C")),
        (("f", "g"), None),
    ]

    expected_cost = 7

    validate_matching(
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


def test_simple_4_branch():
    """
    target graph:

          A
          |
          |
    B<----X---->C
          |
          |
          D

    overcomplete graph:

          a
          |
          b
          |
    c--d--e--f--g
          |
          h
          |
          i

    Matching should be able to match realistic 4 way junction
    """
    target_nodes = ["D", "B", "X", "A", "C"]
    target_edges = [("A", "X"), ("X", "B"), ("X", "C"), ("X", "D")]

    overcomplete_nodes = ["c", "d", "g", "f", "e", "b", "a", "h", "i"]
    overcomplete_edges = [
        ("a", "b"),
        ("b", "e"),
        ("e", "d"),
        ("d", "c"),
        ("e", "f"),
        ("f", "g"),
        ("e", "h"),
        ("h", "i"),
    ]

    node_costs = [
        ("a", "A", 1),
        ("b", "A", 5),
        ("b", "X", 5),
        ("c", "B", 1),
        ("d", "B", 5),
        ("d", "X", 5),
        ("e", "X", 1),
        ("f", "X", 5),
        ("f", "C", 5),
        ("g", "C", 1),
        ("h", "X", 5),
        ("h", "D", 5),
        ("i", "D", 1),
    ]

    edge_costs = [
        (("a", "b"), ("A", "X"), 1),
        (("b", "e"), ("A", "X"), 1),
        (("b", "e"), ("X", "B"), 5),
        (("b", "e"), ("X", "C"), 5),
        (("b", "e"), ("X", "D"), 5),
        (("e", "d"), ("A", "X"), 5),
        (("e", "d"), ("X", "B"), 1),
        (("e", "d"), ("X", "C"), 5),
        (("e", "d"), ("X", "D"), 5),
        (("d", "c"), ("X", "B"), 1),
        (("e", "f"), ("A", "X"), 5),
        (("e", "f"), ("X", "B"), 5),
        (("e", "f"), ("X", "C"), 1),
        (("e", "f"), ("X", "D"), 5),
        (("f", "g"), ("X", "C"), 1),
        (("e", "h"), ("A", "X"), 5),
        (("e", "h"), ("X", "B"), 5),
        (("e", "h"), ("X", "C"), 5),
        (("e", "h"), ("X", "D"), 1),
        (("h", "i"), ("X", "D"), 1),
    ]

    expected_node_matchings = [
        ("e", "X"),
        ("a", "A"),
        ("b", None),
        ("c", "B"),
        ("d", None),
        ("g", "C"),
        ("f", None),
        ("i", "D"),
        ("h", None),
    ]

    expected_edge_matchings = [
        (("a", "b"), ("A", "X")),
        (("b", "e"), ("A", "X")),
        (("e", "d"), ("X", "B")),
        (("d", "c"), ("X", "B")),
        (("e", "f"), ("X", "C")),
        (("f", "g"), ("X", "C")),
        (("e", "h"), ("X", "D")),
        (("h", "i"), ("X", "D")),
    ]

    expected_cost = 13

    validate_matching(
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


def test_confounding_chain():
    """
    target graph:

    A---->B---->C

    overcomplete graph:

     a--b--c--d--e
           |
           f--g--h

    the optimal matching should not assign anything to extra chain
    as long as using it is more expensive than c--d--e
    """

    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b", "c", "d", "e", "f", "g", "h"]
    overcomplete_edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("c", "f"),
        ("f", "g"),
        ("g", "h"),
    ]

    node_costs = [
        ("a", "A", 1),
        ("b", "A", 5),
        ("b", "B", 5),
        ("c", "B", 1),
        ("d", "B", 5),
        ("d", "C", 5),
        ("e", "C", 1),
        ("f", "B", 3),
        ("g", "B", 6),
        ("g", "C", 6),
        ("h", "C", 3),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("b", "c"), ("A", "B"), 1),
        (("b", "c"), ("B", "C"), 5),
        (("c", "d"), ("A", "B"), 5),
        (("c", "d"), ("B", "C"), 1),
        (("d", "e"), ("B", "C"), 1),
        (("c", "f"), ("A", "B"), 5),
        (("c", "f"), ("B", "C"), 5),
        (("f", "g"), ("B", "C"), 3),
        (("g", "h"), ("B", "C"), 3),
    ]

    expected_node_matchings = [
        ("a", "A"),
        ("b", None),
        ("c", "B"),
        ("d", None),
        ("e", "C"),
        ("f", None),
        ("g", None),
        ("h", None),
    ]

    expected_edge_matchings = [
        (("a", "b"), ("A", "B")),
        (("b", "c"), ("A", "B")),
        (("c", "d"), ("B", "C")),
        (("d", "e"), ("B", "C")),
        (("c", "f"), None),
        (("f", "g"), None),
        (("g", "h"), None),
    ]

    expected_cost = 7

    validate_matching(
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


def test_confounding_loop():
    """
    target graph:

    A---->B---->C


    overcomplete graph:

        a--b--c--d--e

            f--g
            | /
            h

    the optimal matching should not match all edges in a loop
    to the same edge to create an "infinite chain".
    """

    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b", "c", "d", "e", "f", "g", "h"]
    overcomplete_edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("f", "g"),
        ("g", "h"),
        ("h", "f"),
    ]

    node_costs = [
        ("a", "A", 1),
        ("b", "A", 5),
        ("b", "B", 5),
        ("c", "B", 1),
        ("d", "B", 5),
        ("d", "C", 5),
        ("e", "C", 1),
        ("f", "B", 2),
        ("g", "B", 6),
        ("g", "C", 6),
        ("h", "B", 6),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("b", "c"), ("A", "B"), 1),
        (("b", "c"), ("B", "C"), 5),
        (("c", "d"), ("A", "B"), 5),
        (("c", "d"), ("B", "C"), 1),
        (("d", "e"), ("B", "C"), 1),
        (("f", "g"), ("B", "C"), 3),
        (("g", "h"), ("B", "C"), 3),
        (("h", "f"), ("B", "C"), 3),
    ]

    expected_node_matchings = [
        ("a", "A"),
        ("b", None),
        ("c", "B"),
        ("d", None),
        ("e", "C"),
        ("f", None),
        ("g", None),
        ("h", None),
    ]

    expected_edge_matchings = [
        (("a", "b"), ("A", "B")),
        (("b", "c"), ("A", "B")),
        (("c", "d"), ("B", "C")),
        (("d", "e"), ("B", "C")),
        (("h", "f"), None),
        (("f", "g"), None),
        (("g", "h"), None),
    ]

    expected_cost = 7

    validate_matching(
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


def test_impossible():
    """
    A--B--C

     a---b
    """

    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b"]
    overcomplete_edges = [
        ("a", "b"),
    ]

    node_costs = [
        ("a", "A", 5),
        ("a", "B", 5),
        ("b", "B", 5),
        ("b", "C", 5),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("a", "b"), ("B", "C"), 1),
    ]

    expected_node_matchings = []

    expected_edge_matchings = []

    expected_cost = float("inf")

    with pytest.raises(ValueError):
        validate_matching(
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


def test_enforced_assignment():
    """
    A--B--C

    a--b--c--d

    enforce the suboptimal assignment of
    b--c--d to A--B--C
    """

    target_nodes = ["A", "B", "C"]
    target_edges = [("A", "B"), ("B", "C")]

    overcomplete_nodes = ["a", "b", "c", "d"]
    overcomplete_edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
    ]

    node_costs = [
        ("a", "A", 1),
        ("b", "A", 5),
        ("b", "B", 1),
        ("c", "B", 5),
        ("c", "C", 1),
        ("d", "C", 5),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("b", "c"), ("A", "B"), 5),
        (("b", "c"), ("B", "C"), 1),
        (("c", "d"), ("B", "C"), 5),
    ]

    expected_node_matchings = [
        ("a", None),
        ("b", "A"),
        ("c", "B"),
        ("d", "C"),
    ]

    expected_edge_matchings = [
        (("a", "b"), None),
        (("b", "c"), ("A", "B")),
        (("c", "d"), ("B", "C")),
    ]

    enforced_assignments = list(
        itertools.chain(expected_node_matchings, expected_edge_matchings[1:])
    )

    expected_cost = 25

    validate_matching(
        target_nodes,
        target_edges,
        overcomplete_nodes,
        overcomplete_edges,
        node_costs,
        edge_costs,
        expected_node_matchings,
        expected_edge_matchings,
        expected_cost,
        enforced_assignments=enforced_assignments,
    )

