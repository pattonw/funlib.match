from neurolight.visualizations.neuroglancer_trees import visualize_trees
from neurolight.transforms.npz_to_graph import parse_npy_graph

import sys
from pathlib import Path
import copy

# filename = Path("mouselight_matchings", "valid", "000")
filename = Path(sys.argv[1])

from funlib.match.preprocess import mouselight_preprocessing
from funlib.match.graph_to_tree_matcher import get_matched

graph = parse_npy_graph(filename / "graph.npz").to_undirected()
tree = parse_npy_graph(filename / "tree.npz")

temp = copy.deepcopy(graph)
mouselight_preprocessing(temp, min_dist=48)

matched = get_matched(temp, tree, "matched", 76)

visualize_trees(
    {
        "graph": graph,
        "tree": tree,
        #"preprocessed": temp,
        "matched": matched
    }
)
