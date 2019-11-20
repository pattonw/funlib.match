import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from collections import Counter
import itertools
from typing import Dict


def mouselight_preprocessing(
    graph: nx.DiGraph(), min_dist: float, voxel_size=[10, 3, 3]
):
    """
    Preprocessing for mouselight skeletonizations s.t. they can be matched
    to mouselight consensus neurons.

    Common problems seen in skeletonizations are:
    
    Gaps:
    A------------------------------B
    a---b---c---d--e      f--g--h--i

    Crossovers:
            A
            |                       a
            |                       |
    B-------C-------D   vs      b---c-d----e
            |                         |
            |                         f
            E

    Solution:
    Remove nodes of degree > 1, and then for all pairwise connected components,
    create edges between all nodes in component a to all nodes in component b
    if the edge is below `min_dist`.
    """

    temp = graph.to_undirected()

    temp.remove_nodes_from(filter(lambda x: temp.degree(x) > 2, list(temp.nodes)))

    wccs = list(list(x) for x in nx.connected_components(temp))
    if len(wccs) < 2:
        return

    nodes = cKDTree([node["location"] for node in temp.nodes.values()])

    spatial_wccs = [cKDTree([temp.nodes[x]["location"] for x in wcc]) for wcc in wccs]
    for (ind_a, spatial_a), (ind_b, spatial_b) in itertools.combinations(
        enumerate(spatial_wccs), 2
    ):
        for node_a_index, closest_nodes in itertools.chain(
            *zip(enumerate(spatial_a.query_ball_tree(spatial_b, min_dist)))
        ):
            for node_b_index in closest_nodes:
                node_a = wccs[ind_a][node_a_index]
                node_b = wccs[ind_b][node_b_index]
                edge_len = np.linalg.norm(
                    temp.nodes[node_a]["location"] - temp.nodes[node_b]["location"]
                )
                if edge_len < min_dist:
                    # Add extra penalties to added edges to minimize cable length
                    # assigned to ambiguous ground truth.
                    graph.add_edge(node_a, node_b, penalty=2*(edge_len + 1))
