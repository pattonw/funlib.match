import networkx as nx
import numpy as np
import pytest

import itertools
import unittest
import logging
from pathlib import Path

from funlib.match.graph_to_tree_matcher import match_graph_to_tree, get_matched
from funlib.match.preprocess import mouselight_preprocessing

logger = logging.getLogger(__name__)


def parse_npy_graph(filename):
    a = np.load(filename)
    edge_list = a["edge_list"]
    node_ids = a["node_ids"]
    locations = a["locations"]
    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)
    graph.add_nodes_from(
        [(nid, {"location": l}) for nid, l in zip(node_ids, locations)]
    )
    return graph

@pytest.mark.slow
class MouselightTest(unittest.TestCase):
    def test_valid(self):
        count = 0
        passed = 0

        mouselight_valid_examples = Path("mouselight_matchings", "valid")
        for example in mouselight_valid_examples.iterdir():
            count += 1
            graph = parse_npy_graph(example / "graph.npz")
            mouselight_preprocessing(graph, min_dist=48)
            tree = parse_npy_graph(example / "tree.npz")
            try:
                matched = get_matched(graph, tree, "matched", 76)
                passed += 1
            except:
                logger.warning(f"Failed on valid {example}")

        self.assertEqual(count, passed)

    def test_invalid(self):
        count = 0
        failed = 0

        mouselight_valid_examples = Path("mouselight_matchings", "invalid")
        for example in mouselight_valid_examples.iterdir():
            count += 1
            graph = parse_npy_graph(example / "graph.npz")
            mouselight_preprocessing(graph, 48)
            tree = parse_npy_graph(example / "tree.npz")
            try:
                match_graph_to_tree(graph, tree, "matched", 76)
                logger.warning(f"Passed on invalid {example}")
            except:
                failed += 1
                pass

        self.assertEqual(count, failed)
