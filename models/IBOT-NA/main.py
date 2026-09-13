import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import time
import json
from collections import defaultdict

import numpy as np
import torch

from args import make_args
from utils import *
from model import FusedGWLoss, IBMLP, anchor_contrastive_loss
from determinism import configure_reproducibility


def set_random_seed(seed, deterministic=True):
    configure_reproducibility(torch, seed=seed, strict=deterministic)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_hparams(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_writer(args):
    if args.no_tensorboard:
        return NullWriter()
    from torch.utils.tensorboard import SummaryWriter
    out_dir = args.log_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return SummaryWriter(save_path(args.dataset, out_dir, args.use_attr))


def ib_schedules(epoch, args, same_dim=True):
    """Return (ib_strength, kl_weight, anchor_weight) for a given epoch."""
    if epoch < args.ib_start_epoch:
        strength = 0.0 if same_dim else 1.0
        return strength, 0.0, 0.0

    t = epoch - args.ib_start_epoch + 1
    if args.ib_strength_warmup_epochs > 0 and same_dim:
        strength = min(1.0, t / float(args.ib_strength_warmup_epochs))
    else:
        strength = 1.0

    if args.ib_kl_anneal_epochs > 0:
        kl_weight = args.ib_kl_weight * min(1.0, t / float(args.ib_kl_anneal_epochs))
    else:
        kl_weight = args.ib_kl_weight

    anchor_weight = args.ib_anchor_weight * strength
    return strength, kl_weight, anchor_weight


def get_dtypes(dtype_name):
    if dtype_name == 'float64':
        return np.float64, torch.float64
    return np.float32, torch.float32


def resolve_dataset_prefix(dataset, ratio):
    """Locate a dataset split in either the code tree or the deliverable data tree."""
    override = os.environ.get('IBOT_NA_DATASET_PREFIX_OVERRIDE', '').strip()
    if override:
        override = os.path.abspath(os.path.expanduser(override))
        expected_path = f'{override}_{ratio:.1f}.npz'
        if not os.path.isfile(expected_path):
            raise FileNotFoundError(
                f"IBOT_NA_DATASET_PREFIX_OVERRIDE does not provide {expected_path}"
            )
        return override
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join('datasets', dataset),
        os.path.join(script_dir, 'datasets', dataset),
        os.path.normpath(os.path.join(script_dir, '..', '..', 'data', dataset)),
    ]
    for prefix in candidates:
        if os.path.exists(f'{prefix}_{ratio:.1f}.npz'):
            return prefix
    return candidates[0]


def _parse_override_value(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(args):
    for override in getattr(args, 'overrides', []) or []:
        if '=' not in override:
            raise ValueError(f"Invalid --override '{override}'. Expected key=value.")
        key, raw_value = override.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --override '{override}'. Empty key.")
        if not hasattr(args, key):
            raise ValueError(f"Unknown override key '{key}'.")
        value = _parse_override_value(raw_value.strip())
        print(f"Override {key} to {value}")
        setattr(args, key, value)


def should_run_full_eval(epoch, args):
    # Evaluate the first and final epochs to guarantee a validation record.
    if epoch == 0 or epoch == args.epochs - 1:
        return True
    if args.eval_every <= 0:
        return False
    return (epoch + 1) % args.eval_every == 0


def format_hits(hits):
    return ", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])


def format_final_metric(value):
    return f"{value:.8f}"


def metric_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def snapshot_metrics(hits, mrr):
    return {key: metric_to_float(value) for key, value in hits.items()}, metric_to_float(mrr)


def _clone_state_dict_to_cpu(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def capture_eval_checkpoint(model, criterion):
    """Capture the exact train-time state selected by validation."""
    prev_s = getattr(criterion, 'prev_s', None)
    if torch.is_tensor(prev_s):
        prev_s = prev_s.detach().cpu().clone()
    return {
        'model_state': _clone_state_dict_to_cpu(model),
        'criterion_state': _clone_state_dict_to_cpu(criterion),
        'criterion_prev_s': prev_s,
    }


def restore_eval_checkpoint(model, criterion, checkpoint, device):
    model_state = {
        key: value.to(device)
        for key, value in checkpoint['model_state'].items()
    }
    criterion_state = {
        key: value.to(device)
        for key, value in checkpoint['criterion_state'].items()
    }
    model.load_state_dict(model_state)
    criterion.load_state_dict(criterion_state)
    prev_s = checkpoint.get('criterion_prev_s')
    criterion.prev_s = prev_s.to(device) if torch.is_tensor(prev_s) else None


def compute_alignment_metrics(dissimilarity, pairs, args):
    if args.fast_metrics:
        return compute_metrics_fast(dissimilarity, pairs)
    if args.metric_batch_size > 0:
        return compute_metrics_batched_sort(dissimilarity, pairs, batch_size=args.metric_batch_size)
    return compute_metrics(dissimilarity, pairs)


def validation_score(hits, mrr, metric_name):
    if metric_name == 'mrr':
        return metric_to_float(mrr)
    if metric_name == 'hits1':
        return metric_to_float(hits[1])
    if metric_name == 'hits10':
        return metric_to_float(hits[10])
    raise ValueError(f"Unknown validation metric: {metric_name}")


def compute_eval_similarity(criterion, eval_out1, eval_out2):
    return criterion.predict(eval_out1, eval_out2, update_cache=False)


if __name__ == '__main__':
    args = make_args()
    set_random_seed(args.seed)
    apply_overrides(args)
    np_dtype, torch_dtype = get_dtypes(args.dtype)

    # Load the fixed split and construct the two input graphs.
    print("Loading data...", end=" ")
    load_attrs = bool(args.use_attr)
    dataset_prefix = resolve_dataset_prefix(args.dataset, args.ratio)
    edge_index1, edge_index2, feat1, feat2, anchor_links, test_pairs, num_nodes1, num_nodes2 = load_data(
        dataset_prefix, args.ratio, load_attrs, dtype=np_dtype)
    anchor_links, val_pairs, test_pairs = split_main_train_val_test(
        anchor_links,
        test_pairs,
        seed=args.seed,
    )
    print(
        f"Split protocol {args.split_protocol}: "
        f"train={anchor_links.shape[0]}, val={val_pairs.shape[0]}, test={test_pairs.shape[0]}, "
        f"seed={args.seed}."
    )
    if val_pairs.shape[0] == 0:
        raise ValueError(
            "A non-empty validation split is required because checkpoint selection is restricted to BEST_VAL."
        )
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]
    G1 = build_nx_graph(edge_index1, anchor1, feat1, num_nodes=num_nodes1)
    G2 = build_nx_graph(edge_index2, anchor2, feat2, num_nodes=num_nodes2)
    print("Done")

    rwr_cache_tag = f"seed{args.seed}_train{anchor_links.shape[0]}_val{val_pairs.shape[0]}"
    rwr1, rwr2 = get_rwr_matrix(G1, G2, anchor_links, args.dataset, args.ratio,
                                dtype=np_dtype, cache_tag=rwr_cache_tag)
    if feat1 is None:
        x1 = rwr1
    else:
        x1 = np.concatenate([feat1, rwr1], axis=1).astype(np_dtype, copy=False)
    if feat2 is None:
        x2 = rwr2
    else:
        x2 = np.concatenate([feat2, rwr2], axis=1).astype(np_dtype, copy=False)
    # Initialize the numerical runtime and graph tensors.
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)
    torch.set_default_dtype(torch_dtype)
    anchor1_t = torch.from_numpy(anchor1).to(torch.long).to(device)
    anchor2_t = torch.from_numpy(anchor2).to(torch.long).to(device)

    G1_tg = build_tg_graph(edge_index1, x1, rwr1, dtype=torch_dtype).to(device)
    G2_tg = build_tg_graph(edge_index2, x2, rwr2, dtype=torch_dtype).to(device)

    n1, n2 = G1_tg.x.shape[0], G2_tg.x.shape[0]
    args.gw_weight = args.alpha / (1 - args.alpha) * min(n1, n2) ** 0.5

    writer = make_writer(args)

    print("Runtime config: "
          f"dtype={args.dtype}, sparse_gw=True, ot_warm_start=True, "
          f"ot_update_every={args.ot_update_every}, eval_every={args.eval_every}, "
          f"in_iter={args.in_iter}, out_iter={args.out_iter}, "
          f"fast_metrics={args.fast_metrics}, "
          f"metric_batch_size={args.metric_batch_size}, "
          f"checkpoint_policy=BEST_VAL, "
          f"val_metric={args.val_metric}")

    eval_hits_list = defaultdict(list)
    eval_mrr_list = []
    threshold_lambda = torch.tensor(args.init_threshold_lambda / (n1 * n2), dtype=torch_dtype, device=device)

    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        model = IBMLP(
            input_dim=G1_tg.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.out_dim,
            ib_dim=args.ib_dim,
            ib_dropout=args.ib_dropout,
            min_logvar=args.ib_min_logvar,
            max_logvar=args.ib_max_logvar,
            free_bits=args.ib_free_bits,
        ).to(device)
        same_dim = args.ib_dim <= 0 or args.ib_dim == args.out_dim
        actual_ib_dim = args.ib_dim if args.ib_dim > 0 else args.out_dim
        print(f"Using IBOT-NA: ib_dim={actual_ib_dim}, "
              f"ib_start_epoch={args.ib_start_epoch}, kl_weight={args.ib_kl_weight}, "
              f"anchor_weight={args.ib_anchor_weight}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FusedGWLoss(
            G1_tg, G2_tg, anchor1, anchor2,
            gw_weight=args.gw_weight,
            gamma_p=args.gamma_p,
            init_threshold_lambda=args.init_threshold_lambda,
            in_iter=args.in_iter,
            out_iter=args.out_iter,
            total_epochs=args.epochs,
        ).to(device)

        print("Training...")
        final_hits = None
        final_mrr = None
        best_selection = None
        final_similarity = None
        best_val_score = -float('inf')

        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            optimizer.zero_grad(set_to_none=True)

            ib_strength, kl_weight, anchor_weight = ib_schedules(epoch, args, same_dim=same_dim)

            out1, out2, ib_kl = model(G1_tg, G2_tg, ib_strength=ib_strength)

            solve_s = epoch % max(1, args.ot_update_every) == 0
            update_lambda = solve_s and ((epoch % max(1, args.lambda_update_every)) == 0)
            ot_loss, _, threshold_lambda = criterion(
                out1=out1,
                out2=out2,
                update_lambda=update_lambda,
                solve_s=solve_s,
            )
            if anchor_weight > 0:
                ib_anchor = anchor_contrastive_loss(out1, out2, anchor1_t, anchor2_t,
                                                    temperature=args.ib_anchor_temperature)
            else:
                ib_anchor = torch.zeros((), dtype=torch_dtype, device=device)

            loss = ot_loss + kl_weight * ib_kl + anchor_weight * ib_anchor
            loss.backward()
            optimizer.step()

            do_eval = should_run_full_eval(epoch, args)
            if do_eval:
                with torch.no_grad():
                    model.eval()
                    eval_out1, eval_out2, eval_kl = model(
                        G1_tg, G2_tg, ib_strength=ib_strength)
                    similarity = compute_eval_similarity(criterion, eval_out1, eval_out2)
                    s_entropy = torch.sum(-similarity * torch.log(similarity + torch.finfo(torch_dtype).eps))
                    hits = None
                    mrr = None
                    val_hits = None
                    val_mrr = None
                    val_hits, val_mrr = compute_alignment_metrics(-similarity, val_pairs, args)
                    score = validation_score(val_hits, val_mrr, args.val_metric)
                    if score > best_val_score:
                        val_snapshot, val_mrr_float = snapshot_metrics(val_hits, val_mrr)
                        best_val_score = score
                        best_selection = {
                            'epoch': epoch + 1,
                            'val_hits': val_snapshot,
                            'val_mrr': val_mrr_float,
                            'score': score,
                            'ib_strength': ib_strength,
                            'checkpoint': capture_eval_checkpoint(model, criterion),
                        }
            else:
                hits, mrr, s_entropy = None, None, None
                val_hits, val_mrr = None, None
                eval_kl = torch.zeros((), dtype=torch_dtype, device=device)

            end = time.time()
            prefix = (f'Epoch {epoch + 1}, Loss: {loss.item():.6f}, OT: {ot_loss.item():.6f}, '
                      f'IB_KL: {ib_kl.item():.6f}, KL_w: {kl_weight:.2e}, '
                      f'IB_strength: {ib_strength:.2f}, AnchorCE: {ib_anchor.item():.6f}, ')

            if do_eval:
                metric_parts = []
                if hits is not None and mrr is not None:
                    metric_parts.append(f'{format_hits(hits)}, MRR: {mrr:.4f}')
                if val_hits is not None and val_mrr is not None:
                    metric_parts.append(f'Val {format_hits(val_hits)}, Val MRR: {val_mrr:.4f}')
                metrics_text = ', '.join(metric_parts) if metric_parts else 'metrics: skipped'
                print(prefix + f's_entropy: {s_entropy:.4f}, threshold_lambda: {threshold_lambda * n1 * n2:.4f}, '
                      f'{metrics_text}, time: {end - start:.2f}s')
            else:
                print(prefix + f'eval: skipped, threshold_lambda: {threshold_lambda * n1 * n2:.4f}, '
                      f'OT_solved: {solve_s}, time: {end - start:.2f}s')

            writer.add_scalar('Loss', loss.item(), epoch)
            writer.add_scalar('OT_Loss', ot_loss.item(), epoch)
            writer.add_scalar('Time/Epoch', end - start, epoch)
            if do_eval:
                writer.add_scalar('S/Entropy', s_entropy, epoch)
                if hits is not None and mrr is not None:
                    writer.add_scalar('MRR', mrr, epoch)
                    for key, value in hits.items():
                        writer.add_scalar(f'Hits/Hits@{key}', value, epoch)
                if val_hits is not None and val_mrr is not None:
                    writer.add_scalar('Val/MRR', val_mrr, epoch)
                    for key, value in val_hits.items():
                        writer.add_scalar(f'Val/Hits@{key}', value, epoch)
            writer.add_scalar('IB/KL', ib_kl.item(), epoch)
            writer.add_scalar('IB/Eval_KL', eval_kl.item(), epoch)
            writer.add_scalar('IB/KL_weight', kl_weight, epoch)
            writer.add_scalar('IB/Strength', ib_strength, epoch)
            writer.add_scalar('IB/AnchorCE', ib_anchor.item(), epoch)
            writer.add_scalar('IB/Anchor_weight', anchor_weight, epoch)

        if best_selection is None:
            raise RuntimeError("No validation metrics were produced for BEST_VAL checkpoint selection.")
        restore_eval_checkpoint(model, criterion, best_selection['checkpoint'], device)
        with torch.no_grad():
            model.eval()
            eval_out1, eval_out2, _ = model(
                G1_tg, G2_tg, ib_strength=best_selection['ib_strength'])
            similarity = criterion.predict(eval_out1, eval_out2, update_cache=False)
            test_hits, test_mrr = compute_alignment_metrics(-similarity, test_pairs, args)
            final_hits, final_mrr = snapshot_metrics(test_hits, test_mrr)
            final_similarity = similarity.detach().cpu()
        print(
            f"BEST_VAL Epoch {best_selection['epoch']}, "
            f"Val Hits@1: {format_final_metric(best_selection['val_hits'][1])}, "
            f"Val Hits@10: {format_final_metric(best_selection['val_hits'][10])}, "
            f"Val MRR: {format_final_metric(best_selection['val_mrr'])}, "
            f"Test Hits@1: {format_final_metric(final_hits[1])}, "
            f"Test Hits@10: {format_final_metric(final_hits[10])}, "
            f"Test MRR: {format_final_metric(final_mrr)}"
        )
        if args.save_final_alignment and run == args.runs - 1:
            alignment_path = os.path.abspath(os.path.expanduser(args.save_final_alignment))
            if final_similarity is None:
                raise RuntimeError("Saving the final alignment requires clean evaluation with validation selection.")
            os.makedirs(os.path.dirname(alignment_path) or ".", exist_ok=True)
            torch.save({
                'similarity': final_similarity,
                'train_pairs': torch.from_numpy(anchor_links.copy()),
                'val_pairs': torch.from_numpy(val_pairs.copy()),
                'test_pairs': torch.from_numpy(test_pairs.copy()),
                'seed': int(args.seed),
                'best_epoch': int(best_selection['epoch']) if best_selection is not None else None,
            }, alignment_path)
            print(f"Saved final alignment to {alignment_path}")
        for key, value in final_hits.items():
            eval_hits_list[key].append(value)
        eval_mrr_list.append(final_mrr)
        print("")

    eval_hits = {}
    eval_hits_std = {}
    for key, value in eval_hits_list.items():
        hits_list = np.array([val for val in value])
        eval_hits[key] = hits_list.mean()
        eval_hits_std[key] = hits_list.std()
    eval_mrr = np.array(eval_mrr_list).mean()
    eval_mrr_std = np.array(eval_mrr_list).std()

    hparam_dict = {
        'dataset': args.dataset,
        'use_attr': args.use_attr,
        'epochs': args.epochs,
        'split_protocol': args.split_protocol,
        'lr': args.lr,
        'alpha': args.alpha,
        'gamma_p': args.gamma_p,
        'threshold_lambda': threshold_lambda.detach().cpu().item() if torch.is_tensor(threshold_lambda) else threshold_lambda,
        'ib_dim': args.ib_dim if args.ib_dim > 0 else args.out_dim,
        'ib_kl_weight': args.ib_kl_weight,
        'ib_anchor_weight': args.ib_anchor_weight,
        'ib_start_epoch': args.ib_start_epoch,
        'ib_sampling_evaluation': 'posterior_mean',
        'dtype': args.dtype,
        'sparse_gw': True,
        'ot_update_every': args.ot_update_every,
        'eval_every': args.eval_every,
        'checkpoint_policy': 'BEST_VAL',
        'val_metric': args.val_metric,
    }
    writer.add_hparams(hparam_dict, {'hparam/MRR': eval_mrr,
                                     'hparam/std_MRR': eval_mrr_std,
                                     **{f'hparam/Hits@{key}': value for key, value in eval_hits.items()},
                                     **{f'hparam/std_Hits@{key}': value for key, value in eval_hits_std.items()}})
    writer.close()
