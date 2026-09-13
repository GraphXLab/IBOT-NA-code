import numpy as np
import networkx as nx
import os
from pathlib import Path
import tempfile
import torch
from tqdm import tqdm


def _environment_cache_tag():
    value = os.environ.get('IBOT_NA_RWR_CACHE_TAG', '').strip()
    if not value:
        return None
    return ''.join(character if character.isalnum() or character in {'-', '_'} else '_' for character in value)


def _force_recompute_rwr():
    return os.environ.get('IBOT_NA_FORCE_RECOMPUTE_RWR', '').strip().casefold() in {'1', 'true', 'yes', 'on'}


def _rwr_cache_root():
    configured = os.environ.get('IBOT_NA_RWR_CACHE_ROOT', '').strip()
    return Path(configured) if configured else Path(tempfile.gettempdir()) / 'ibot-na' / 'rwr'


def get_rwr_matrix(G1, G2, anchor_links, dataset, ratio, dtype=np.float32, cache_tag=None):
    """
    Get distance matrix of the network
    :param G1: input graph 1
    :param G2: input graph 2
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param dtype: data type
    :return: distance matrix (num of nodes x num of anchor nodes)
    """
    cache_root = _rwr_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    tags = [str(tag) for tag in (cache_tag, _environment_cache_tag()) if tag]
    suffix = f'_{"_".join(tags)}' if tags else ''
    rwr_path = cache_root / f'rwr_emb_{dataset}_{ratio:.1f}{suffix}_sparse.npz'
    if rwr_path.exists() and not _force_recompute_rwr():
        print(f"Loading RWR scores from {rwr_path}...", end=" ")
        data = np.load(rwr_path)
        rwr1, rwr2 = data['rwr1'], data['rwr2']
        print("Done")
    else:
        if _force_recompute_rwr():
            print(f"Forcing RWR recomputation for {rwr_path}...", end=" ")
        rwr1, rwr2 = rwr_scores(G1, G2, anchor_links, dtype)
        print(f"Saving RWR scores to {rwr_path}...", end=" ")
        np.savez(rwr_path, rwr1=rwr1, rwr2=rwr2)
        print("Done")

    return rwr1, rwr2


def rwr_scores(G1, G2, anchor_links, dtype=np.float32):
    """
    Compute initial node embedding vectors by random walk with restart
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :param dtype: data type
    :return: rwr_score1, rwr_score2: RWR vectors of the networks
    """

    rwr_score1 = rwr_score(G1, anchor_links[:, 0], desc="Computing RWR scores for G1", dtype=dtype)
    rwr_score2 = rwr_score(G2, anchor_links[:, 1], desc="Computing RWR scores for G2", dtype=dtype)

    return rwr_score1, rwr_score2


def rwr_score_networkx(G, anchors, restart_prob=0.15, desc='Computing RWR scores', dtype=np.float32):
    """
    Random walk with restart for a single graph
    :param G: network G, i.e., networkx graph
    :param anchors: anchor nodes
    :param restart_prob: restart probability
    :param desc: description for tqdm
    :param dtype: data type
    :return: rwr: rwr vectors of the network
    """

    n = G.number_of_nodes()
    rwr = np.zeros((n, len(anchors))).astype(dtype)

    for i, node in enumerate(tqdm(anchors, desc=desc)):
        s = nx.pagerank(G, personalization={node: 1}, alpha=1-restart_prob)
        for k, v in s.items():
            rwr[k, i] = v

    return rwr


def rwr_score(G, anchors, restart_prob=0.15, desc='Computing batched sparse RWR scores',
              dtype=np.float32, max_iters=1000, tol=1e-6, batch_size=512):
    """
    Compute batched sparse random-walk-with-restart scores.

    Calling NetworkX PageRank once per anchor is exact but inefficient for
    datasets containing thousands of supervised anchors.
    """
    n = G.number_of_nodes()
    anchors = np.asarray(anchors, dtype=np.int64)
    rwr = np.zeros((n, len(anchors)), dtype=dtype)
    if len(anchors) == 0:
        return rwr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

    adj = nx.to_scipy_sparse_array(
        G,
        nodelist=np.arange(n),
        dtype=np.float64 if dtype == np.float64 else np.float32,
        weight='weight',
        format='coo',
    )
    row = torch.from_numpy(adj.row.astype(np.int64, copy=False)).to(device)
    col = torch.from_numpy(adj.col.astype(np.int64, copy=False)).to(device)
    val = torch.from_numpy(adj.data.astype(dtype, copy=False)).to(device=device, dtype=torch_dtype)

    deg = torch.zeros(n, dtype=torch_dtype, device=device)
    deg.scatter_add_(0, row, val)
    keep = deg[row] > 0
    trans_indices = torch.stack([col[keep], row[keep]], dim=0)
    trans_values = val[keep] / deg[row[keep]]
    trans_mat = torch.sparse_coo_tensor(trans_indices, trans_values, (n, n), device=device).coalesce()
    anchor_tensor = torch.from_numpy(anchors).to(device=device, dtype=torch.long)

    with torch.no_grad():
        for start in tqdm(range(0, len(anchors), batch_size), desc=desc):
            end = min(start + batch_size, len(anchors))
            batch_anchors = anchor_tensor[start:end]
            width = end - start
            landmark_vecs = torch.zeros((n, width), dtype=torch_dtype, device=device)
            landmark_vecs[batch_anchors, torch.arange(width, device=device)] = 1
            rwr_vecs = torch.ones((n, width), dtype=torch_dtype, device=device)
            for _ in range(max_iters):
                old = rwr_vecs
                rwr_vecs = (1 - restart_prob) * torch.sparse.mm(trans_mat, rwr_vecs) + restart_prob * landmark_vecs
                if torch.max(torch.abs(rwr_vecs - old)).item() < tol:
                    break
            rwr[:, start:end] = rwr_vecs.cpu().numpy().astype(dtype, copy=False)

    return rwr
