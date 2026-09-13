import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def _infer_num_nodes(edge_index, pairs, pair_col, x=None):
    candidates = []
    if edge_index.size:
        candidates.append(int(edge_index.max()) + 1)
    if pairs.size:
        candidates.append(int(pairs[:, pair_col].max()) + 1)
    if x is not None:
        candidates.append(int(x.shape[0]))
    return max(candidates) if candidates else 0


def load_data(dataset, p, use_attr, dtype=np.float32):
    """
    Load dataset.
    :param dataset: dataset name
    :param p: training ratio
    :param use_attr: whether to use input node attributes
    :param dtype: data type
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        x1, x2: input node attributes of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
        num_nodes1, num_nodes2: explicit graph sizes
    """

    data = np.load(f'{dataset}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    if use_attr and 'x1' in data.files and 'x2' in data.files:
        x1, x2 = data['x1'].astype(dtype), data['x2'].astype(dtype)
    else:
        x1, x2 = None, None

    pairs = np.concatenate([anchor_links, test_pairs], axis=0)
    num_nodes1 = int(data['num_nodes1']) if 'num_nodes1' in data.files else _infer_num_nodes(edge_index1, pairs, 0, x1)
    num_nodes2 = int(data['num_nodes2']) if 'num_nodes2' in data.files else _infer_num_nodes(edge_index2, pairs, 1, x2)

    return edge_index1, edge_index2, x1, x2, anchor_links, test_pairs, num_nodes1, num_nodes2


def split_train_validation_pairs(anchor_links, val_ratio=0.5, seed=0):
    """
    Deterministically split training anchor links into train and validation.

    Validation pairs are used only for model selection. At least one training
    pair is kept whenever two or more anchors are available.
    """
    anchor_links = np.asarray(anchor_links, dtype=np.int64)
    if anchor_links.size == 0 or val_ratio <= 0.0:
        return anchor_links, np.empty((0, 2), dtype=np.int64)

    n_pairs = anchor_links.shape[0]
    if n_pairs < 2:
        return anchor_links, np.empty((0, 2), dtype=np.int64)

    val_size = int(round(n_pairs * float(val_ratio)))
    val_size = max(1, min(val_size, n_pairs - 1))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_pairs, generator=generator)
    val_idx = torch.sort(perm[:val_size]).values.numpy()
    train_idx = torch.sort(perm[val_size:]).values.numpy()
    return anchor_links[train_idx], anchor_links[val_idx]


def split_main_train_val_test(pos_pairs, test_pairs, seed=0):
    """
    Build the formal 10/10/80 protocol from a stored 20/80 split.

    The `pos_pairs` array is the 20% supervised anchor pool and `test_pairs` is
    the untouched 80% test set. Only the supervised pool is divided into equal
    training and validation subsets.
    """
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64)
    test_pairs = np.asarray(test_pairs, dtype=np.int64)
    train_pairs, val_pairs = split_train_validation_pairs(
        pos_pairs,
        val_ratio=0.5,
        seed=seed,
    )
    return train_pairs, val_pairs, test_pairs


def build_nx_graph(edge_index, anchor_nodes, x=None, num_nodes=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param anchor_nodes: anchor nodes
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    if num_nodes is None and x is not None:
        num_nodes = x.shape[0]
    if num_nodes is not None:
        G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(edge_index)
    G.x = x
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G.anchor_nodes = anchor_nodes
    return G


def build_tg_graph(edge_index, x, rwr, dtype=torch.float32):
    """
    Build a PyG Data object from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param x: node attributes of the graph
    :param rwr: random walk with restart scores
    :param dtype: data type
    :return: a PyG Data object
    """

    edge_index_tensor = torch.from_numpy(edge_index.T).to(torch.int64)
    x_tensor = torch.from_numpy(x).to(dtype)
    data = Data(x=x_tensor, edge_index=edge_index_tensor)
    data.rwr = torch.from_numpy(rwr).to(dtype)
    return data
