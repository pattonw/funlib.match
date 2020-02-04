# TODO: Create invalid examples that target each specific constraint.


def not_overcomplete():
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
        ("a", None, 0),
        ("b", None, 0),
    ]

    edge_costs = [
        (("a", "b"), ("A", "B"), 1),
        (("a", "b"), ("B", "C"), 1),
    ]

    return (
        target_nodes,
        target_edges,
        overcomplete_nodes,
        overcomplete_edges,
        node_costs,
        edge_costs,
    )
