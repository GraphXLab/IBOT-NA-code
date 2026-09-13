import torch
import os
import numpy as np


def compute_distance_matrix(emb1, emb2):
    """
    Compute distance matrix between two sets of embeddings
    :param emb1: node embeddings of graph 1
    :param emb2: node embeddings of graph 2
    :return: distance matrix
    """

    emb1 = emb1 / torch.linalg.norm(emb1, ord=2, axis=1, keepdims=True)
    emb2 = emb2 / torch.linalg.norm(emb2, ord=2, axis=1, keepdims=True)
    dists = 1 - emb1 @ emb2.T

    return dists


def compute_metrics_ltr(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    distances = dissimilarity[test_pairs[:, 0]]
    device = dissimilarity.device

    hits = {}
    ranks = torch.argsort(distances, dim=1)
    test_pairs = torch.from_numpy(test_pairs).to(torch.int64).to(device)
    signal_hit = ranks == test_pairs[:, 1].view(-1, 1)
    for k in hit_top_ks:
        hits[k] = torch.sum(signal_hit[:, :k]) / test_pairs.shape[0]

    mrr = torch.mean(1 / (torch.where(ranks == test_pairs[:, 1].view(-1, 1))[1] + 1))

    return hits, mrr


def compute_metrics_rtl(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    distances = dissimilarity.T[test_pairs[:, 1]]
    device = dissimilarity.device

    hits = {}
    ranks = torch.argsort(distances, dim=1)
    test_pairs = torch.from_numpy(test_pairs).to(torch.int64).to(device)
    signal_hit = ranks == test_pairs[:, 0].view(-1, 1)
    for k in hit_top_ks:
        hits[k] = torch.sum(signal_hit[:, :k]) / test_pairs.shape[0]

    mrr = torch.mean(1 / (torch.where(ranks == test_pairs[:, 0].view(-1, 1))[1] + 1))

    return hits, mrr


def compute_metrics(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """
    Compute bidirectional Hits@K and MRR.

    Report the arithmetic mean of graph1-to-graph2 and graph2-to-graph1 scores.
    :param dissimilarity: dissimilarity matrix (n1 x n2)
    :param test_pairs: test pairs
    :param hit_top_ks: list of k for HITS@k
    :return:
        hits: HITS@k
        mrr: MRR
    """

    distances1 = dissimilarity[test_pairs[:, 0]]
    distances2 = dissimilarity.T[test_pairs[:, 1]]
    device = dissimilarity.device

    hits = {}

    ranks1 = torch.argsort(distances1, dim=1)
    ranks2 = torch.argsort(distances2, dim=1)

    test_pairs = torch.from_numpy(test_pairs).to(torch.int64).to(device)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    for k in hit_top_ks:
        hits_ltr = torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]
        hits_rtl = torch.sum(signal2_hit[:, :k]) / test_pairs.shape[0]
        hits[k] = (hits_ltr + hits_rtl) / 2

    mrr_ltr = torch.mean(1 / (torch.where(ranks1 == test_pairs[:, 1].view(-1, 1))[1] + 1))
    mrr_rtl = torch.mean(1 / (torch.where(ranks2 == test_pairs[:, 0].view(-1, 1))[1] + 1))
    mrr = (mrr_ltr + mrr_rtl) / 2

    return hits, mrr


def _batched_sort_ranks(dissimilarity, src, tgt, batch_size, left_to_right=True):
    positions = []
    for start in range(0, src.numel(), batch_size):
        end = min(start + batch_size, src.numel())
        src_batch = src[start:end]
        tgt_batch = tgt[start:end]
        if left_to_right:
            distances = dissimilarity[src_batch]
            target = tgt_batch
        else:
            distances = dissimilarity[:, tgt_batch].T
            target = src_batch
        ranks = torch.argsort(distances, dim=1)
        positions.append(torch.where(ranks == target.view(-1, 1))[1] + 1)
        del distances, ranks
    return torch.cat(positions, dim=0)


def compute_metrics_batched_sort(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100), batch_size=256):
    """
    Full-argsort metrics computed in batches to reduce peak memory.

    This keeps the same per-row/per-column argsort and bidirectional mean
    semantics as compute_metrics, but avoids materializing all test rows and
    all sorted ranks at once.
    """
    device = dissimilarity.device
    test_pairs_t = torch.as_tensor(test_pairs, dtype=torch.long, device=device)
    src = test_pairs_t[:, 0]
    tgt = test_pairs_t[:, 1]
    batch_size = max(1, int(batch_size))

    ranks_ltr = _batched_sort_ranks(dissimilarity, src, tgt, batch_size, left_to_right=True)
    ranks_rtl = _batched_sort_ranks(dissimilarity, src, tgt, batch_size, left_to_right=False)

    hits = {}
    for k in hit_top_ks:
        hits_ltr = torch.mean((ranks_ltr <= k).to(torch.float32))
        hits_rtl = torch.mean((ranks_rtl <= k).to(torch.float32))
        hits[k] = (hits_ltr + hits_rtl) / 2

    mrr_ltr = torch.mean(1.0 / ranks_ltr.to(torch.float32))
    mrr_rtl = torch.mean(1.0 / ranks_rtl.to(torch.float32))
    mrr = (mrr_ltr + mrr_rtl) / 2
    return hits, mrr


def save_path(dataset, out_dir, use_attr=False):
    if dataset == 'ACM-DBLP':
        dataset = 'ACM-DBLP_attr' if use_attr else 'ACM-DBLP'

    if not os.path.exists(f'{out_dir}'):
        os.makedirs(f'{out_dir}')
    if not os.path.exists(f'{out_dir}/{dataset}_results'):
        os.makedirs(f'{out_dir}/{dataset}_results')
    runs = len([f for f in os.listdir(f'{out_dir}/{dataset}_results') if os.path.isdir(f'{out_dir}/{dataset}_results/{f}')])
    runs_str = str(runs).zfill(3)
    return f'{out_dir}/{dataset}_results/run_{runs_str}'


def compute_metrics_fast(dissimilarity, test_pairs, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """
    Compute exact bidirectional rank metrics without a full argsort.

    The full metric implementation sorts every test row and column, costing
    O(|test| n log n). This implementation counts candidates with smaller
    dissimilarity in O(|test| n) and averages both alignment directions. Values
    match full sorting except for exact ties, which use optimistic tie handling.
    """
    device = dissimilarity.device
    test_pairs_t = torch.as_tensor(test_pairs, dtype=torch.long, device=device)
    src = test_pairs_t[:, 0]
    tgt = test_pairs_t[:, 1]

    rows = dissimilarity[src]
    true_row_scores = rows.gather(1, tgt.view(-1, 1))
    ranks_ltr = 1 + torch.sum(rows < true_row_scores, dim=1)

    cols = dissimilarity.T[tgt]
    true_col_scores = cols.gather(1, src.view(-1, 1))
    ranks_rtl = 1 + torch.sum(cols < true_col_scores, dim=1)

    hits = {}
    for k in hit_top_ks:
        hits_ltr = torch.mean((ranks_ltr <= k).to(torch.float32))
        hits_rtl = torch.mean((ranks_rtl <= k).to(torch.float32))
        hits[k] = (hits_ltr + hits_rtl) / 2

    mrr_ltr = torch.mean(1.0 / ranks_ltr.to(torch.float32))
    mrr_rtl = torch.mean(1.0 / ranks_rtl.to(torch.float32))
    mrr = (mrr_ltr + mrr_rtl) / 2
    return hits, mrr
