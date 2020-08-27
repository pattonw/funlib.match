import networkx as nx
import numpy as np

import itertools
import pytest

from funlib.match.graph_to_tree_matcher import GraphToTreeMatcher

from .realistic import large

from funlib.match.get_constraints_reference import (
    get_constraints as ref_all,
    get_121_t2g_node_constraint as ref_0,
    get_12n_t2g_edge_constraint as ref_1,
    get_g2t_node_constraint as ref_2,
    get_branch_topology_constraint as ref_3,
    get_solid_chain_constraint as ref_4,
    get_degree_constraint as ref_5,
)
from funlib.match.get_constraints import (
    get_constraints as get_all,
    get_121_t2g_node_constraint as get_0,
    get_12n_t2g_edge_constraint as get_1,
    get_g2t_node_constraint as get_2,
    get_branch_topology_constraint as get_3,
    get_solid_chain_constraint as get_4,
    get_degree_constraint as get_5,
)


def get_inputs(m, constraint, implementation):
    if constraint == -1:
        if implementation == "reference":
            return [m.graph, m.tree, m.node_match_costs, m.edge_match_costs]
        if implementation == "vectorized":
            return [m.graph, m.tree, m.node_match_costs, m.edge_match_costs]
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 0:
        if implementation == "reference":
            return [m.tree.nodes(), m.t2g_match_indicators]
        if implementation == "vectorized":
            node_match_costs = m.node_match_costs

            tree_nodes = [int(x) for x in m.tree.nodes()]
            node_indx_map = {x: i for i, x in enumerate(tree_nodes)}

            node_matchings = [
                (u, node_indx_map[v] if v is not None else len(tree_nodes) + 1)
                for u, v, c in node_match_costs
            ]

            return [node_matchings, len(tree_nodes)]
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 1:
        if implementation == "reference":
            return [m.tree.edges(), m.t2g_match_indicators]
        if implementation == "vectorized":

            edge_match_costs = m.edge_match_costs

            tree_edges = [(int(u), int(v)) for u, v in m.tree.edges()]
            edge_indx_map = {x: i for i, x in enumerate(tree_edges)}

            edge_matchings = [(u, edge_indx_map[v]) for u, v, c in edge_match_costs]
            return [edge_matchings, len(tree_edges), len(m.node_match_costs)]
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 2:
        if implementation == "reference":
            return [m.graph.nodes(), m.g2t_match_indicators]
        if implementation == "vectorized":
            node_match_costs = m.node_match_costs

            graph_nodes = [int(x) for x in m.graph.nodes()]
            node_indx_map = {x: i for i, x in enumerate(graph_nodes)}

            node_matchings = [(node_indx_map[u], v) for u, v, c in node_match_costs]

            return [node_matchings, len(graph_nodes)]
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 3:
        if implementation == "reference":
            return [
                m.graph.nodes(),
                m.g2t_match_indicators,
                m.tree.out_edges,
                m.tree.in_edges,
                m.graph.out_edges,
                m.graph.in_edges,
            ]
        if implementation == "vectorized":
            node_match_costs = m.node_match_costs

            graph_nodes = [int(x) for x in m.graph.nodes()]
            graph_node_indx_map = {x: i for i, x in enumerate(graph_nodes)}

            tree_nodes = [int(x) for x in m.tree.nodes()]
            tree_node_indx_map = {x: i for i, x in enumerate(tree_nodes)}

            edge_match_costs = m.edge_match_costs
            graph_edges = [(int(u), int(v)) for u, v in m.graph.edges()]
            graph_edge_indx_map = {x: i for i, x in enumerate(graph_edges)}

            tree_edges = [(int(u), int(v)) for u, v in m.tree.edges()]
            tree_edge_indx_map = {x: i for i, x in enumerate(tree_edges)}

            node_matchings = [
                (
                    graph_node_indx_map[u],
                    tree_node_indx_map[v] if v is not None else len(tree_nodes),
                )
                for u, v, c in node_match_costs
            ]
            edge_matching_indices = {
                (graph_edge_indx_map[a], tree_edge_indx_map[b]): i
                for i, (a, b, _) in enumerate(edge_match_costs)
            }

            in_g_edges = {
                graph_node_indx_map[n]: [
                    graph_edge_indx_map[x] for x in m.graph.in_edges(n)
                ]
                for n in graph_nodes
            }
            out_g_edges = {
                graph_node_indx_map[n]: [
                    graph_edge_indx_map[x] for x in m.graph.out_edges(n)
                ]
                for n in graph_nodes
            }
            in_t_edges = {
                tree_node_indx_map[n]: [
                    tree_edge_indx_map[x] for x in m.tree.in_edges(n)
                ]
                for n in tree_nodes
            }
            out_t_edges = {
                tree_node_indx_map[n]: [
                    tree_edge_indx_map[x] for x in m.tree.out_edges(n)
                ]
                for n in tree_nodes
            }

            return [
                node_matchings,
                in_g_edges,
                out_g_edges,
                in_t_edges,
                out_t_edges,
                edge_matching_indices,
            ]
            raise NotImplementedError()
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 4:
        if implementation == "reference":
            return [
                m.graph.nodes,
                m.g2t_match_indicators,
                m.tree.edges,
                m.graph.in_edges,
                m.graph.out_edges,
            ]
        if implementation == "vectorized":
            node_match_costs = m.node_match_costs

            graph_nodes = [int(x) for x in m.graph.nodes()]
            graph_node_indx_map = {x: i for i, x in enumerate(graph_nodes)}

            tree_nodes = [int(x) for x in m.tree.nodes()]
            tree_node_indx_map = {x: i for i, x in enumerate(tree_nodes)}

            edge_match_costs = m.edge_match_costs

            node_match_indx = {
                (
                    graph_node_indx_map[u],
                    tree_node_indx_map[v] if v is not None else len(tree_nodes),
                ): i
                for i, (u, v, _) in enumerate(node_match_costs)
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
            return [
                len(graph_nodes),
                edge_matchings,
                node_match_indx,
                len(node_match_costs),
            ]
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 5:
        if implementation == "reference":
            return [
                m.graph.nodes,
                m.g2t_match_indicators,
                m.tree.degree,
                {
                    u: list(
                        itertools.chain(m.graph.successors(u), m.graph.predecessors(u))
                    )
                    for u in m.graph.nodes
                },
            ]
        if implementation == "vectorized":
            node_match_costs = m.node_match_costs

            graph_nodes = [int(x) for x in m.graph.nodes()]
            graph_node_indx_map = {x: i for i, x in enumerate(graph_nodes)}

            tree_nodes = [int(x) for x in m.tree.nodes()]
            tree_node_indx_map = {x: i for i, x in enumerate(tree_nodes)}

            tree_degree = {int(x): m.tree.degree(x) for x in tree_nodes}

            edge_match_costs = m.edge_match_costs

            node_matchings = [
                (
                    graph_node_indx_map[u],
                    tree_node_indx_map[v] if v is not None else len(tree_nodes),
                )
                for u, v, _ in node_match_costs
            ]
            edge_matchings = {
                (
                    graph_node_indx_map[g_u],
                    graph_node_indx_map[g_v],
                    tree_node_indx_map[t_u],
                    tree_node_indx_map[t_v],
                ): i
                for i, ((g_u, g_v), (t_u, t_v), _) in enumerate(edge_match_costs)
            }
            return [node_matchings, tree_degree, edge_matchings, tree_nodes]
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()


def get_func(constraint, implementation):
    if constraint == -1:
        if implementation == "reference":
            return ref_all
        if implementation == "vectorized":
            return get_all
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 0:
        if implementation == "reference":
            return ref_0
        if implementation == "vectorized":
            return get_0
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 1:
        if implementation == "reference":
            return ref_1
        if implementation == "vectorized":
            return get_1
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 2:
        if implementation == "reference":
            return ref_2
        if implementation == "vectorized":
            return get_2
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 3:
        if implementation == "reference":
            return ref_3
        if implementation == "vectorized":
            return get_3
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 4:
        if implementation == "reference":
            return ref_4
        if implementation == "vectorized":
            return get_4
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()
    if constraint == 5:
        if implementation == "reference":
            return ref_5
        if implementation == "vectorized":
            return get_5
        if implementation == "cython":
            raise NotImplementedError()
        if implementation == "cython-vectorized":
            raise NotImplementedError()
        if implementation == "rusty":
            raise NotImplementedError()


@pytest.mark.benchmark(group="get_constraints",)
@pytest.mark.parametrize(
    "data_func", [large,],
)
@pytest.mark.parametrize("constraint", [-1, 5])
@pytest.mark.parametrize(
    "implementation", ["reference", "vectorized"],
)
def test_constraint_init(benchmark, data_func, constraint, implementation):
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

    overcomplete = nx.Graph()
    overcomplete.add_nodes_from(overcomplete_nodes)
    overcomplete.add_edges_from(overcomplete_edges)

    m = GraphToTreeMatcher(
        overcomplete,
        target,
        node_costs,
        edge_costs,
        use_gurobi=False,
        create_constraints=False,
    )

    inputs = get_inputs(m, constraint, implementation)
    func = get_func(constraint, implementation)

    benchmark(func, *inputs)
