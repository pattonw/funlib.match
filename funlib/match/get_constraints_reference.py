import itertools


def get_constraints(graph, tree, node_match_costs, edge_match_costs):
    # initialize any necessary variables
    tree_nodes = tree.nodes
    graph_nodes = graph.nodes
    tree_edges = tree.edges
    graph_edges = graph.edges
    tree_out_edges = tree.out_edges
    graph_out_edges = graph.out_edges
    tree_in_edges = tree.in_edges
    graph_in_edges = graph.in_edges
    tree_degree = tree.degree
    graph_neighbors = {
        u: list(itertools.chain(graph.successors(u), graph.predecessors(u)))
        for u in graph.nodes
    }

    g2ts = {}
    t2gs = {}
    for i, (source, target, cost) in enumerate(
        itertools.chain(node_match_costs, edge_match_costs)
    ):

        u2t_node_indicators = g2ts.setdefault(source, {})
        u2t_node_indicators[target] = i
        v2g_node_indicators = t2gs.setdefault(target, {})
        v2g_node_indicators[source] = i

    # get constraints
    all_constraints = {}
    all_constraints["1-1 T-G NODES"] = get_121_t2g_node_constraint(tree_nodes, t2gs)
    all_constraints["1-N T-G EDGES"] = get_12n_t2g_edge_constraint(tree_edges, t2gs)
    all_constraints["G-T NODES"] = get_g2t_node_constraint(graph_nodes, g2ts)
    all_constraints["BRANCH TOPOLOGY"] = get_branch_topology_constraint(
        graph_nodes,
        g2ts,
        tree_out_edges,
        tree_in_edges,
        graph_out_edges,
        graph_in_edges,
    )
    all_constraints["CHAIN"] = get_solid_chain_constraint(
        graph_nodes,
        g2ts,
        tree_edges,
        graph_in_edges,
        graph_out_edges,
    )
    all_constraints["DEGREE"] = get_degree_constraint(
        graph_nodes,
        g2ts,
        tree_degree,
        graph_neighbors,
    )
    # return constraints
    return all_constraints


def get_121_t2g_node_constraint(tree_nodes, t2gs):
    """
    (2b) Every node in T must match to exactly one node in G
    """
    constraints = []
    for tree_n in tree_nodes:
        new_constraint = []
        for match_indicator in t2gs.get(tree_n, {}).values():
            new_constraint.append((match_indicator, 1))
        constraints.append(new_constraint)
    return constraints, "Equal", 1


def get_12n_t2g_edge_constraint(tree_edges, t2gs):
    """
    modified
    (2c): Every edge in T must match to **at least one** edge in G

    For strict subgraph isomorphisms, this would be exactly 1,
        but we want to allow "chains"
    """
    constraints = []
    for tree_e in tree_edges:

        new_constraint = []
        for match_indicator in t2gs.get(tree_e, {}).values():
            new_constraint.append((match_indicator, -1))
        constraints.append(new_constraint)
    return constraints, "LessEqual", -1


def get_g2t_node_constraint(graph_nodes, g2ts):
    """
    modified
    (2d) Every node in G must match

    A possible match for every node in G is None
    """
    constraints = []
    for graph_n in graph_nodes:
        new_constraint = []
        for match_indicator in g2ts.get(graph_n, {}).values():
            new_constraint.append((match_indicator, 1))
        constraints.append(new_constraint)
    return constraints, "Equal", 1


def get_branch_topology_constraint(
    graph_nodes,
    g2ts,
    tree_out_edges,
    tree_in_edges,
    graph_out_edges,
    graph_in_edges,
):
    """
    For every pair of matched nodes j, k in V(G), V(T),
        (2e) Every edge targeting k in T must match to an edge targeting j in G
        (2f) Every edge originating from k in T must match to an edge originating from j in G
    """
    constraints = []
    for graph_n in graph_nodes:
        tree_nodes = [x for x in g2ts.get(graph_n, {}).keys() if x is not None]
        for tree_n in tree_nodes:
            node_match_indicator = g2ts.get(graph_n, {}).get(tree_n)

            # (2e)
            for tree_out_e in tree_out_edges(tree_n):
                new_constraint = []
                new_constraint.append((node_match_indicator, 1))
                for graph_out_e in graph_out_edges(graph_n):
                    g2t_e_indicator = g2ts.get(graph_out_e, {}).get(tree_out_e)
                    if g2t_e_indicator is not None:
                        new_constraint.append((g2t_e_indicator, -1))
                constraints.append(new_constraint)

            # (2f)
            for tree_in_e in tree_in_edges(tree_n):
                new_constraint = []
                new_constraint.append((node_match_indicator, 1))
                for graph_in_e in graph_in_edges(graph_n):
                    g2t_e_indicator = g2ts.get(graph_in_e, {}).get(tree_in_e)
                    if g2t_e_indicator is not None:
                        new_constraint.append((g2t_e_indicator, -1))
                constraints.append(new_constraint)
    return constraints, "LessEqual", 0


def get_solid_chain_constraint(
    graph_nodes, g2ts, tree_edges, graph_in_edges, graph_out_edges
):
    """
    EXTRA CONSTRAINT, NOT FROM PAPER:
    The previous constraints are enough to cover isomorphisms, but we need to model chains
    in G representing edges in S
    Consider:

    T:      A---------------------------------------B
    G:      a--b--c---------------------------d--e--f

    Under previous constraints, matching AB to ab and ef would be sufficient.

    Consider a node i in G and an edge kl in T.
    If i matches to k, then we know
    there should be exactly 1 edge originating from i that matches to kl, and 0
    edges targeting i should match to kl.
    Similarly if i matches to l, then there should be exactly 1 edge targeting i
    that matches to kl, and 0 edges originating from i should match to kl.
    Finally, if i matches to neither k nor l, then to maintain a chain, we must
    have exactly 1 edge originating from i match to kl *and* exactly 1 edge
    targeting i to match to kl.
    """
    constraints = []

    for graph_n in graph_nodes():

        possible_edges = set(
            tree_e for tree_n in g2ts.get(graph_n, {}) for tree_e in tree_edges(tree_n)
        )
        for tree_e in possible_edges:
            new_constraint = []
            for graph_in_e in graph_in_edges(graph_n):
                indicator = g2ts.get(graph_in_e, {}).get(tree_e)
                if indicator is not None:
                    # -1 if an in edge matches
                    new_constraint.append((indicator, -1))
            for graph_out_e in graph_out_edges(graph_n):
                indicator = g2ts.get(graph_out_e, {}).get(tree_e)
                if indicator is not None:
                    # +1 if an out edge matches
                    new_constraint.append((indicator, 1))

            for tree_n, tree_n_indicator in g2ts.get(graph_n, {}).items():
                # tree_e must be an out edge
                if tree_n == tree_e[0]:
                    new_constraint.append((tree_n_indicator, -1))
                # tree_e must be an in edge
                if tree_n == tree_e[1]:
                    new_constraint.append((tree_n_indicator, 1))

            constraints.append(new_constraint)

    return constraints, "Equal", 0


def get_degree_constraint(graph_nodes, g2ts, tree_degree, graph_neighbors):
    """
    Now that we allow chains, it is possible that multiple
    chains pass through the same node, or a chain might pass through
    a branch point. To avoid this we add a degree constraint.

    given x_ij for all i in V(G) and j in V(T)
    given y_ab_cd for all (a, b) in E(G) and (c, d) in E(T)
    let degree(None) = 2
    For every node i in V(G)
        let N = SUM(degree(c)*x_ic) for all c in V(T) Union None
        let y = SUM(y_ai_cd) + SUM(y_ia_cd)
            for all a adjacent to i, and all (c,d) in E(T)
        y - N <= 0
    """
    constraints = []
    for graph_n in graph_nodes():
        new_constraint = []
        for tree_n, tree_n_indicator in g2ts.get(graph_n, {}).items():
            d = 2 if tree_n is None else tree_degree(tree_n)
            new_constraint.append((tree_n_indicator, -d))
        for neighbor in graph_neighbors[graph_n]:
            for adj_edge_indicator in g2ts.get((graph_n, neighbor), {}).values():
                new_constraint.append((adj_edge_indicator, 1))
            for adj_edge_indicator in g2ts.get((neighbor, graph_n), {}).values():
                new_constraint.append((adj_edge_indicator, 1))
        constraints.append(new_constraint)
    return constraints, "LessEqual", 0
