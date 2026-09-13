from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    packaged_datasets = [
        'foursquare-twitter', 'phone-email', 'arxiv', 'GGI',
        'Douban', 'ACM-DBLP', 'dbp15k_zh-en',
    ]
    dataset_choices = sorted(set(packaged_datasets + [f'{name}-pa' for name in packaged_datasets]))
    parser.add_argument('--dataset', dest='dataset', type=str, default='phone-email',
                        choices=dataset_choices,
                        help='dataset name, with an optional -pa suffix')
    parser.add_argument('--seed', dest='seed', type=int, default=123, help='random seed')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.2,
                        choices=[0.2], help='training ratio: 0.2')
    parser.add_argument('--split_protocol', dest='split_protocol', type=str, default='train10_val10_test80',
                        choices=['train10_val10_test80'],
                        help='alignment split protocol; the formal protocol divides the supervised pool equally between training and validation')
    parser.add_argument('--use_attr', dest='use_attr', default=False, action='store_true',
                        help='use input node attributes')

    # Device / numerical settings
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu', help='use GPU')
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32', choices=['float32', 'float64'],
                        help='numeric dtype; float32 is recommended for routine GPU and CPU experiments')

    # Model settings
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--out_dim', dest='out_dim', type=int, default=128, help='output dimension')
    # Variational bottleneck settings
    parser.add_argument('--ib_dim', dest='ib_dim', type=int, default=0,
                        help='latent dimension; 0 uses out_dim')
    parser.add_argument('--ib_kl_weight', dest='ib_kl_weight', type=float, default=1e-6,
                        help='final weight of KL(q(z|h)||N(0,I))')
    parser.add_argument('--ib_kl_anneal_epochs', dest='ib_kl_anneal_epochs', type=int, default=50,
                        help='linearly warm up the IB KL weight after ib_start_epoch')
    parser.add_argument('--ib_start_epoch', dest='ib_start_epoch', type=int, default=30,
                        help='number of warm-up epochs before activating the information bottleneck')
    parser.add_argument('--ib_strength_warmup_epochs', dest='ib_strength_warmup_epochs', type=int, default=30,
                        help='linearly increase bottleneck interpolation strength after ib_start_epoch')
    parser.add_argument('--ib_dropout', dest='ib_dropout', type=float, default=0.0,
                        help='dropout before the bottleneck')
    parser.add_argument('--ib_min_logvar', dest='ib_min_logvar', type=float, default=-8.0,
                        help='minimum log-variance for the variational bottleneck')
    parser.add_argument('--ib_max_logvar', dest='ib_max_logvar', type=float, default=4.0,
                        help='maximum log-variance for the variational bottleneck')
    parser.add_argument('--ib_free_bits', dest='ib_free_bits', type=float, default=0.0,
                        help='free nats per latent dimension before KL is penalized')
    parser.add_argument('--ib_anchor_weight', dest='ib_anchor_weight', type=float, default=0.02,
                        help='supervised anchor InfoNCE weight')
    parser.add_argument('--ib_anchor_temperature', dest='ib_anchor_temperature', type=float, default=0.2,
                        help='temperature for the optional anchor InfoNCE loss')
    # Loss / OT settings
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.9, help='weight of gw distance')
    parser.add_argument('--gamma_p', dest='gamma_p', type=float, default=1e-2, help='entropy regularization parameter')
    parser.add_argument('--in_iter', dest='in_iter', type=int, default=5, help='number of Sinkhorn inner iterations')
    parser.add_argument('--out_iter', dest='out_iter', type=int, default=10, help='number of proximal outer iterations')
    parser.add_argument('--ot_update_every', dest='ot_update_every', type=int, default=1,
                        help='solve OT every k epochs and reuse the previous transport plan between updates')
    parser.add_argument('--lambda_update_every', dest='lambda_update_every', type=int, default=1,
                        help='update threshold lambda every k epochs; update is skipped on non-OT-update epochs')

    # Training / evaluation settings
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')
    parser.add_argument('--eval_every', dest='eval_every', type=int, default=5,
                        help='run deterministic OT validation every k epochs')
    parser.add_argument('--val_metric', dest='val_metric', type=str, default='mrr',
                        choices=['mrr', 'hits1', 'hits10'],
                        help='validation metric used for checkpoint selection')
    parser.add_argument('--fast_metrics', dest='fast_metrics', default=True, action='store_true',
                        help='compute Hits/MRR by rank counting instead of full argsort')
    parser.add_argument('--sort_metrics', dest='fast_metrics', action='store_false',
                        help='use old full-argsort metrics implementation')
    parser.add_argument('--metric_batch_size', dest='metric_batch_size', type=int, default=0,
                        help='batch size for full-argsort metrics; 0 materializes all test rows/columns')
    parser.add_argument('--no_tensorboard', dest='no_tensorboard', default=False, action='store_true',
                        help='disable TensorBoard logging to reduce I/O overhead')
    parser.add_argument('--log_dir', dest='log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--save_final_alignment', dest='save_final_alignment', type=str, default='',
                        help='optional .pt path for the final selected transport matrix and split pairs')
    parser.add_argument('--override', dest='overrides', action='append', default=[],
                        help='apply key=value overrides after loading settings/*.json; can be repeated')

    # Experiment settings
    parser.add_argument('--init_threshold_lambda', dest='init_threshold_lambda', type=float, default=1.0,
                        help='initial sampling threshold (lambda)')

    args = parser.parse_args()
    return args
