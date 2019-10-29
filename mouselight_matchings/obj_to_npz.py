import numpy as np
import networkx as nx

from tqdm import tqdm

from pathlib import Path
import pickle

consensus_pickles = Path(".")


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


for pickled_failure in tqdm(consensus_pickles.iterdir()):
    if pickled_failure.name.endswith(".obj"):
        f_id = pickled_failure.name.split("_")[0]
        Path(f_id).mkdir(exist_ok=True)

        data = pickle.load(pickled_failure.open("rb"))
        graph = data["graph"]
        if len(graph.nodes) < 2 or len(graph.edges) < 2:
            continue
        tree = data["tree"]
        if len(tree.nodes) < 2 or len(tree.edges) < 2:
            continue
        component = data.get("component")
        if component is not None:
            comp = tree.subgraph(component)
            if len(comp.nodes) < 2 or len(comp.edges) < 2:
                continue

        g_edge_list = np.array(graph.edges, dtype=int)
        g_node_ids = np.array(graph.nodes, dtype=int)
        g_locations = np.array([graph.nodes[nid]["location"] for nid in g_node_ids])

        np.savez(
            f"{f_id}/graph.npz",
            edge_list=g_edge_list,
            node_ids=g_node_ids,
            locations=g_locations,
        )

        if component is not None:
            comp = tree.subgraph(component)
            t_edge_list = np.array(comp.edges, dtype=int)
            t_node_ids = np.array(comp.nodes, dtype=int)
            t_locations = np.array([comp.nodes[nid]["location"] for nid in t_node_ids])
        else:
            t_edge_list = np.array(tree.edges, dtype=int)
            t_node_ids = np.array(tree.nodes, dtype=int)
            t_locations = np.array([tree.nodes[nid]["location"] for nid in t_node_ids])

        np.savez(
            f"{f_id}/tree.npz",
            edge_list=t_edge_list,
            node_ids=t_node_ids,
            locations=t_locations,
        )

