import functools
import itertools
from typing import List, Tuple, Dict


def get_constraints(graph, tree, node_match_costs, edge_match_costs):
    # initialize any necessary variables
    node_match_costs = node_match_costs

    graph_nodes = [x for x in graph.nodes()]
    num_graph_nodes = len(graph_nodes)
    graph_node_indx_map = {x: i for i, x in enumerate(graph_nodes)}

    tree_nodes = [x for x in tree.nodes()]
    num_tree_nodes = len(tree_nodes)
    tree_node_indx_map = {x: i for i, x in enumerate(tree_nodes)}

    graph_edges = [(u, v) for u, v in graph.edges()]
    graph_edge_indices = [
        (graph_node_indx_map[u], graph_node_indx_map[v]) for u, v in graph_edges
    ]
    num_graph_edges = len(graph_edges)
    graph_edge_indx_map = {x: i for i, x in enumerate(graph_edges)}

    tree_edges = [(u, v) for u, v in tree.edges()]
    tree_edge_indices = [
        (tree_node_indx_map[u], tree_node_indx_map[v]) for u, v in tree_edges
    ]
    num_tree_edges = len(tree_edges)
    tree_edge_indx_map = {x: i for i, x in enumerate(tree_edges)}

    node_matchings = [
        (
            graph_node_indx_map[u],
            tree_node_indx_map[v] if v is not None else len(tree_nodes),
        )
        for u, v, c in node_match_costs
    ]
    node_match_indx = {
        (
            graph_node_indx_map[u],
            tree_node_indx_map[v] if v is not None else len(tree_nodes),
        ): i
        for i, (u, v, _) in enumerate(node_match_costs)
    }
    num_node_matchings = len(node_matchings)
    edge_matching_indices = {
        (graph_edge_indx_map[a], tree_edge_indx_map[b]): i
        for i, (a, b, _) in enumerate(edge_match_costs)
    }

    edge_matchings = {
        (
            graph_node_indx_map[g_u],
            graph_node_indx_map[g_v],
            tree_node_indx_map[t_u],
            tree_node_indx_map[t_v],
        ): i
        for i, ((g_u, g_v), (t_u, t_v), _) in enumerate(edge_match_costs)
    }

    in_g_edges = {
        graph_node_indx_map[n]: [graph_edge_indx_map[x] for x in graph.in_edges(n)]
        for n in graph_nodes
    }
    out_g_edges = {
        graph_node_indx_map[n]: [graph_edge_indx_map[x] for x in graph.out_edges(n)]
        for n in graph_nodes
    }
    in_t_edges = {
        tree_node_indx_map[n]: [tree_edge_indx_map[x] for x in tree.in_edges(n)]
        for n in tree_nodes
    }
    out_t_edges = {
        tree_node_indx_map[n]: [tree_edge_indx_map[x] for x in tree.out_edges(n)]
        for n in tree_nodes
    }
    tree_degree = {tree_node_indx_map[x]: tree.degree(x) for x in tree_nodes}
    tree_degrees = [tree_degree[i] for i in range(len(tree_nodes))]

    # get constraints
    all_constraints = {}
    all_constraints["1-1 T-G NODES"] = get_121_t2g_node_constraint(
        node_matchings, num_tree_nodes
    )
    all_constraints["1-N T-G EDGES"] = get_12n_t2g_edge_constraint(
        edge_matching_indices, num_tree_edges, num_node_matchings
    )
    all_constraints["G-T NODES"] = get_g2t_node_constraint(
        node_matchings, num_graph_nodes
    )
    # all_constraints["BRANCH TOPOLOGY"] = get_branch_topology_constraint_v2(
    #     node_matchings,
    #     in_g_edges,
    #     out_g_edges,
    #     in_t_edges,
    #     out_t_edges,
    #     edge_matching_indices,
    # )

    all_constraints["BRANCH TOPOLOGY"] = get_branch_topology_constraint_v2(
        edge_matching_indices,
        node_matchings,
        tree_degrees,
        graph_edge_indices,
        tree_edge_indices,
        node_match_indx,
        num_node_matchings,
    )

    all_constraints["CHAIN"] = get_solid_chain_constraint(
        num_graph_nodes, edge_matchings, node_match_indx, num_node_matchings
    )
    all_constraints["DEGREE"] = get_degree_constraint(
        node_matchings, tree_degree, edge_matchings, tree_nodes
    )
    # return constraints
    return all_constraints


def get_121_t2g_node_constraint(node_matchings, num_nodes):
    constraints = [[] for _ in range(num_nodes)]
    for i, (_, t_n_index) in enumerate(node_matchings):
        if t_n_index < num_nodes:
            constraints[t_n_index].append((i, 1))
    return constraints, "Equal", 1


def get_12n_t2g_edge_constraint(edge_matchings, num_edges, num_node_matchings):
    """
    modified
    (2c): Every edge in T must match to **at least one** edge in G

    For strict subgraph isomorphisms, this would be exactly 1,
        but we want to allow "chains"
    """
    constraints = [[] for _ in range(num_edges)]

    for i, (_, t_e_index) in enumerate(edge_matchings):
        constraints[t_e_index].append((i + num_node_matchings, -1))
    return constraints, "LessEqual", -1


def get_g2t_node_constraint(node_matchings, num_nodes):
    """
    modified
    (2d) Every node in G must match

    A possible match for every node in G is None
    """
    constraints = [[] for _ in range(num_nodes)]
    for i, (g_n_index, _) in enumerate(node_matchings):
        constraints[g_n_index].append((i, 1))
    return constraints, "Equal", 1


def get_branch_topology_constraint(
    node_matchings,
    in_g_edges,
    out_g_edges,
    in_t_edges,
    out_t_edges,
    edge_matching_indices,
):
    """
    For every pair of matched nodes j, k in V(G), V(T),
        (2e) Every edge targeting k in T must match to an edge targeting j in G
        (2f) Every edge originating from k in T must match to an edge originating from j in G
    """
    constraints = {}
    for u, v in node_matchings:
        u_constraints = constraints.setdefault((u, v), [])

    for i, (g_n, t_n) in enumerate(node_matchings):
        in_g_es, in_t_es = in_g_edges[g_n], in_t_edges.get(t_n, [])
        out_g_es, out_t_es = out_g_edges[g_n], out_t_edges.get(t_n, [])

        for in_t in in_t_es:
            new_constraint = []
            new_constraint.append((i, 1))
            for in_g in in_g_es:
                in_g_t_index = edge_matching_indices.get((in_g, in_t))
                if in_g_t_index is not None:
                    new_constraint.append((in_g_t_index + len(node_matchings), -1))
            constraints[(g_n, t_n)].append(new_constraint)

        for out_t in out_t_es:
            new_constraint = []
            new_constraint.append((i, 1))
            for out_g in out_g_es:
                out_g_t_index = edge_matching_indices.get((out_g, out_t))
                if out_g_t_index is not None:
                    new_constraint.append((out_g_t_index + len(node_matchings), -1))
            constraints[(g_n, t_n)].append(new_constraint)

    return (
        [
            constraint
            for constraints in constraints.values()
            for constraint in constraints
        ],
        "LessEqual",
        0,
    )


def get_branch_topology_constraint_v2(
    edge_matchings: List[Tuple[int, int]],
    node_matchings: List[Tuple[int, int]],
    tree_degrees: List[int],
    graph_edges: Dict[int, Tuple[int, int]],
    tree_edges: Dict[int, Tuple[int, int]],
    node_match_indices: Dict[Tuple[int, int], int],
    num_node_matchings: int,
):
    """
    For every pair of matched nodes j, k in V(G), V(T),
        (2e) Every edge targeting k in T must match to an edge targeting j in G
        (2f) Every edge originating from k in T must match to an edge originating from j in G

    """
    constraints = {}

    for i, (g_e, t_e) in enumerate(edge_matchings):
        g_u, g_v = graph_edges[g_e]
        t_u, t_v = tree_edges[t_e]

        uu_index = node_match_indices.get((g_u, t_u))
        vv_index = node_match_indices.get((g_v, t_v))

        if uu_index is not None:
            uu_constraints = constraints.setdefault((g_u, t_u), {})

            uu_ge_constraint = uu_constraints.setdefault(t_e, [(uu_index, 1)])

            uu_ge_constraint.append((i + num_node_matchings, -1))

        if vv_index is not None:
            vv_constraints = constraints.setdefault((g_v, t_v), {})

            vv_ge_constraint = vv_constraints.setdefault(
                t_e, [(node_match_indices[(g_v, t_v)], 1)]
            )

            vv_ge_constraint.append((i + num_node_matchings, -1))

    for (g_n, t_n) in node_matchings:
        if t_n == len(tree_degrees):
            continue
        node_match_constraints = constraints.setdefault((g_n, t_n), {})
        if len(node_match_constraints) < tree_degrees[t_n]:
            node_match_constraints[None] = [(node_match_indices[(g_n, t_n)], 1)]

    return (
        [
            edge_constraint
            for node_match_constraints in constraints.values()
            for edge_constraint in node_match_constraints.values()
        ],
        "LessEqual",
        0,
    )


def get_solid_chain_constraint(
    num_g_nodes, edge_matchings, node_match_indx, num_node_matchings
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

    DIFFERENCE TO REFERENCE IMPLEMENTATION:
        Original method:
        get g_n
        get all possible t_e that match to any g_e adjacent to g_n
        for each g_n, t_e create a constraint:

            for each every g_in -> t_e match -1
            for each every g_out -> t_e match +1
            for every possible t_n match of g_n:
                (g_n, t_n) -1 if t_n = t_e.source
                (g_n, t_n) +1 if t_n = t_e.target


        Reference: simple 4 branch (from tests):
        g_n = h
        t_e = XB
        g_ins -> t_e: (eh, XB) -1
        g_outs -> t_e:
        g_n -> t_n: (h, X) -1  <- THIS IS THE DIFFERENCE

        Reformulated: simple 4 branch (from tests):
        g_u, g_v, t_u, t_v = e, h, X, B

        Why are we missing hX in reformulation?
        Only add node balance if g_u==t_u or g_v==t_v
        Thus since there is no (he, XB) we never check hX

        Is it correct to keep it?
        Yes
        When will it occur?
        a matching (g_n, t_n) exists, but no (gu, gv, tu, tv) exists
        s.t. (g_n, t_n) == (g_u, t_u) or (g_n, t_n) == (g_v, t_v).

        Do any of the other constraints prevent (g_n, t_n) in this case?
        Yes: Degree constraint and topology constraint and existance of all t_n and t_e
        all guarantee that it would be impossible to match a (g_n, t_n) without
        any adjacent edges. 

    """
    constraints = [{} for _ in range(num_g_nodes)]

    for i, (g_u, g_v, t_u, t_v) in enumerate(edge_matchings):
        u_constraint = constraints[g_u].setdefault((t_u, t_v), set([]))
        v_constraint = constraints[g_v].setdefault((t_u, t_v), set([]))

        # (t_u, t_v) get +1 if matches to out edge of g_u
        u_constraint.add((i + num_node_matchings, 1))
        # (t_u, t_v) get -1 if matches to in edge of g_v
        v_constraint.add((i + num_node_matchings, -1))

        # if g_u matches t_u, then we balance g_u for this edge matching
        uu = node_match_indx.get((g_u, t_u))
        if uu is not None:
            u_constraint.add((uu, -1))

        # if g_v matches t_v, then we balance g_v for this edge matching
        vv = node_match_indx.get((g_v, t_v))
        if vv is not None:
            v_constraint.add((vv, 1))

    return (
        [
            list(constraint)
            for node_constraints in constraints
            for constraint in node_constraints.values()
        ],
        "Equal",
        0,
    )


def get_degree_constraint(node_matchings, tree_degree, edge_matchings, tree_nodes):
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
    constraints = {}
    for i, (g_n, t_n) in enumerate(node_matchings):
        g_n_degree_constraint = constraints.setdefault(g_n, [])
        g_n_degree_constraint.append((i, -tree_degree.get(t_n, 2)))

    for i, (g_u, g_v, t_u, t_v) in enumerate(edge_matchings):
        constraints.get(g_u, []).append((i + len(node_matchings), 1))
        constraints.get(g_v, []).append((i + len(node_matchings), 1))

    return list(constraints.values()), "LessEqual", 0
