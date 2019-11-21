from funlib.match.helper_functions import match

import networkx as nx


def test_build_matched():
    """
    target graph:

       A---->B---->C

    overcomplete graph:

    a--b--c--d--e--f--g
    """
    target = nx.DiGraph()
    target.add_nodes_from(["A", "B", "C"])
    target.add_edges_from([("A", "B"), ("B", "C")])

    overcomplete = nx.Graph()
    overcomplete.add_nodes_from(["a", "b", "c", "d", "e", "f", "g"])
    overcomplete.add_edges_from(
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"),]
    )

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

    matched = match(overcomplete, target, node_costs, edge_costs, use_gurobi=True)
    matched2 = match(overcomplete, target, node_costs, edge_costs, use_gurobi=False)

    for node, m in expected_node_matchings:
        if m is not None:
            assert node in matched.nodes() and node in matched2.nodes()
            
    for edge, m in expected_edge_matchings:
        if m is None:
            assert edge not in matched.edges() and edge not in matched2.edges()
        else:
            assert edge in matched.edges() and edge in matched2.edges()
