import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalInformationBottleneck(nn.Module):
    """
    Stabilized variational information bottleneck q_phi(z|h).

    IBOT-NA uses the deterministic posterior mean as its transport-cost
    representation.
    """

    def __init__(self, input_dim, latent_dim, min_logvar=-8.0, max_logvar=4.0, free_bits=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.free_bits = float(free_bits)
        self.reset_parameters()

    def reset_parameters(self):
        if self.input_dim == self.latent_dim:
            nn.init.eye_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
        else:
            nn.init.xavier_uniform_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h):
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=self.min_logvar, max=self.max_logvar)

        kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_dim = torch.clamp(kl_dim - self.free_bits, min=0.0)
        kl = torch.sum(kl_dim, dim=1).mean()
        z = F.normalize(mu, p=2, dim=1, eps=1e-12)
        return z, kl


class IBMLP(torch.nn.Module):
    """
    Residual MLP encoder with a stabilized variational information bottleneck.

    input -> shared residual MLP -> normalized h
          -> q_phi(z|h) -> deterministic z=mu by default -> OT cost.

    If latent_dim == output_dim, ib_strength smoothly interpolates between
    """

    def __init__(self, input_dim, hidden_dim, output_dim, ib_dim=None, ib_dropout=0.0,
                 min_logvar=-8.0, max_logvar=4.0, free_bits=0.0):
        super(IBMLP, self).__init__()
        ib_dim = output_dim if ib_dim is None or ib_dim <= 0 else ib_dim
        self.output_dim = int(output_dim)
        self.ib_dim = int(ib_dim)
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(input_dim + hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(ib_dropout)
        self.bottleneck = VariationalInformationBottleneck(
            input_dim=output_dim,
            latent_dim=self.ib_dim,
            min_logvar=min_logvar,
            max_logvar=max_logvar,
            free_bits=free_bits,
        )

    def encode_pre_bottleneck(self, x):
        h = torch.cat([x, self.act(self.lin1(x))], dim=1)
        h = self.lin2(h)
        h = self.dropout(h)
        h = F.normalize(h, p=2, dim=1, eps=1e-12)
        return h

    def forward(
        self,
        G1_data,
        G2_data,
        ib_strength=1.0,
    ):
        h1 = self.encode_pre_bottleneck(G1_data.x)
        h2 = self.encode_pre_bottleneck(G2_data.x)
        z1, kl1 = self.bottleneck(h1)
        z2, kl2 = self.bottleneck(h2)

        ib_strength = float(ib_strength)
        if self.ib_dim == self.output_dim and ib_strength < 1.0:
            z1 = F.normalize((1.0 - ib_strength) * h1 + ib_strength * z1, p=2, dim=1, eps=1e-12)
            z2 = F.normalize((1.0 - ib_strength) * h2 + ib_strength * z2, p=2, dim=1, eps=1e-12)

        ib_kl = 0.5 * (kl1 + kl2)
        return z1, z2, ib_kl


class FusedGWLoss(torch.nn.Module):
    def __init__(self, G1_tg, G2_tg, anchor1, anchor2, gw_weight=20, gamma_p=1e-2,
                 init_threshold_lambda=1, in_iter=5, out_iter=10, lambda_step=5e-2,
                 total_epochs=250):
        super().__init__()
        self.device = G1_tg.x.device
        self.dtype = G1_tg.x.dtype
        self.gw_weight = gw_weight
        self.gamma_p = gamma_p
        self.in_iter = in_iter
        self.out_iter = out_iter
        self.lambda_step = lambda_step
        self.total_epochs = total_epochs
        self.prev_s = None

        self.n1, self.n2 = G1_tg.num_nodes, G2_tg.num_nodes
        threshold_init = max(float(init_threshold_lambda), 0.0) / (self.n1 * self.n2)
        self.register_buffer('threshold_lambda', torch.tensor(threshold_init,
                                                              dtype=self.dtype, device=self.device))
        self.register_buffer('max_threshold_lambda', torch.tensor(threshold_init,
                                                                  dtype=self.dtype, device=self.device))
        self.register_buffer('marginal_a', torch.ones(self.n1, dtype=self.dtype, device=self.device) / self.n1)
        self.register_buffer('marginal_b', torch.ones(self.n2, dtype=self.dtype, device=self.device) / self.n2)
        self.register_buffer('log_marginal_a', torch.log(self.marginal_a))
        self.register_buffer('log_marginal_b', torch.log(self.marginal_b))

        self.edge_index1 = G1_tg.edge_index.to(self.device)
        self.edge_index2 = G2_tg.edge_index.to(self.device)

    def _dense_inter_cost(self, out1, out2):
        return torch.exp(-(out1 @ out2.T))

    def _edge_values(self, out, edge_index):
        src, dst = edge_index[0], edge_index[1]
        emb_vals = torch.exp(-torch.sum(out[src] * out[dst], dim=1))
        return emb_vals

    def _sparse_intra_cost(self, out, edge_index, n):
        vals = self._edge_values(out, edge_index)
        return torch.sparse_coo_tensor(edge_index, vals, (n, n), dtype=self.dtype, device=self.device).coalesce()

    @staticmethod
    def _sparse_square(sp_tensor):
        sp_tensor = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(sp_tensor.indices(), sp_tensor.values().pow(2), sp_tensor.shape,
                                       dtype=sp_tensor.dtype, device=sp_tensor.device).coalesce()

    def compute_costs(self, out1, out2):
        inter_c = self._dense_inter_cost(out1, out2)
        intra_c1 = self._sparse_intra_cost(out1, self.edge_index1, self.n1)
        intra_c2 = self._sparse_intra_cost(out2, self.edge_index2, self.n2)
        intra_c1_sq = self._sparse_square(intra_c1)
        intra_c2_sq = self._sparse_square(intra_c2)
        return inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq

    def solve_ot(self, inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq, update_cache=True):
        init_s = self.prev_s
        s = sinkhorn_stable(inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq,
                            gw_weight=self.gw_weight,
                            gamma_p=self.gamma_p,
                            threshold_lambda=self.threshold_lambda,
                            in_iter=self.in_iter,
                            out_iter=self.out_iter,
                            marginal_a=self.marginal_a,
                            marginal_b=self.marginal_b,
                            log_marginal_a=self.log_marginal_a,
                            log_marginal_b=self.log_marginal_b,
                            init_s=init_s)
        if update_cache:
            self.prev_s = s.detach()
        return s

    def forward(self, out1, out2, update_lambda=True, solve_s=True):
        inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq = self.compute_costs(out1, out2)

        with torch.no_grad():
            if solve_s or self.prev_s is None:
                s = self.solve_ot(inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq, update_cache=True)
            else:
                s = self.prev_s.detach()
            if update_lambda:
                new_lambda = self.update_lambda(inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq, s)
                self._update_threshold_lambda(new_lambda)

        s_hat = s - self.threshold_lambda
        w_loss = torch.sum(inter_c * s_hat)
        gw_loss = self._gw_loss(intra_c1, intra_c2, intra_c1_sq, intra_c2_sq, s_hat)
        loss = w_loss + self.gw_weight * gw_loss + 20
        return loss, s, self.threshold_lambda

    def _gw_loss(self, c1, c2, c1_sq, c2_sq, s_hat):
        a_hat = torch.sum(s_hat, dim=1)
        b_hat = torch.sum(s_hat, dim=0)
        left = _left_matvec(c1_sq, a_hat).view(-1, 1)
        right = _row_vec_matmul(b_hat, c2_sq).view(1, -1)
        cross = _left_right_cost(c1, s_hat, c2)
        return torch.sum((left + right - 2.0 * cross) * s_hat)

    @torch.no_grad()
    def _update_threshold_lambda(self, proposed_lambda):
        proposed = torch.as_tensor(proposed_lambda, dtype=self.dtype, device=self.device).reshape(())
        self.threshold_lambda.clamp_(min=0.0, max=self.max_threshold_lambda)
        if not bool(torch.isfinite(proposed).item()):
            return

        bounded = torch.clamp(proposed.detach(), min=0.0, max=self.max_threshold_lambda)
        self.threshold_lambda.mul_(1.0 - self.lambda_step).add_(self.lambda_step * bounded)
        self.threshold_lambda.clamp_(min=0.0, max=self.max_threshold_lambda)

    @torch.no_grad()
    def predict(self, out1, out2, update_cache=False):
        """Compute deterministic OT alignment without updating lambda."""
        inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq = self.compute_costs(out1, out2)
        return self.solve_ot(inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq, update_cache=update_cache)

    def update_lambda(self, inter_c, c1, c2, c1_sq, c2_sq, s):
        # This factorization is algebraically equivalent to the dense formulation
        # and avoids materializing the n1 x n2 intermediate matrix.
        k1 = torch.sum(inter_c)
        s_row = torch.sum(s, dim=1)
        s_col = torch.sum(s, dim=0)

        c1_row_sum = _row_sum(c1, self.n1)
        c2_row_sum = _row_sum(c2, self.n2)
        c1_sq_row_sum = _row_sum(c1_sq, self.n1)
        c2_sq_col_sum = _col_sum(c2_sq, self.n2)

        term1 = self.n2 * torch.sum(c1_sq_row_sum * s_row)
        term2 = self.n1 * torch.sum(c2_sq_col_sum * s_col)
        term3 = 2.0 * torch.sum((c1_row_sum.view(1, -1) @ s).view(-1) * c2_row_sum)
        k2 = term1 + term2 - term3

        k3 = (self.n2 ** 2) * torch.sum(c1_sq_row_sum) + (self.n1 ** 2) * torch.sum(c2_sq_col_sum) \
             - 2.0 * torch.sum(c1_row_sum) * torch.sum(c2_row_sum)
        eps = torch.finfo(k3.dtype).eps
        return (k1 + 2 * self.gw_weight * k2) / (2 * self.gw_weight * (k3 + eps))


# Numerical helper functions

def _left_matvec(mat, vec):
    return torch.sparse.mm(mat, vec.view(-1, 1)).view(-1)


def _row_vec_matmul(vec, mat):
    return torch.sparse.mm(mat.transpose(0, 1).coalesce(), vec.view(-1, 1)).view(-1)


def _left_right_cost(c1, s_hat, c2):
    left = torch.sparse.mm(c1, s_hat)
    return torch.sparse.mm(c2, left.T).T


def _row_sum(mat, n_rows):
    mat = mat.coalesce()
    out = torch.zeros(n_rows, dtype=mat.dtype, device=mat.device)
    out.index_add_(0, mat.indices()[0], mat.values())
    return out


def _col_sum(mat, n_cols):
    mat = mat.coalesce()
    out = torch.zeros(n_cols, dtype=mat.dtype, device=mat.device)
    out.index_add_(0, mat.indices()[1], mat.values())
    return out


def anchor_contrastive_loss(out1, out2, anchor1, anchor2, temperature=0.2):
    """
    Alignment relevance term for the bottleneck.

    It tells the bottleneck which alignment information to preserve.
    """
    if anchor1.numel() == 0:
        return out1.new_tensor(0.0)
    logits_12 = (out1[anchor1] @ out2.T) / temperature
    logits_21 = (out2[anchor2] @ out1.T) / temperature
    loss_12 = F.cross_entropy(logits_12, anchor2)
    loss_21 = F.cross_entropy(logits_21, anchor1)
    return 0.5 * (loss_12 + loss_21)


def sinkhorn_stable(inter_c, intra_c1, intra_c2, intra_c1_sq, intra_c2_sq, threshold_lambda,
                    in_iter=5, out_iter=10, gw_weight=20, gamma_p=1e-2,
                    marginal_a=None, marginal_b=None, log_marginal_a=None, log_marginal_b=None,
                    init_s=None):
    n1, n2 = inter_c.shape
    device, dtype = inter_c.device, inter_c.dtype

    if marginal_a is None:
        marginal_a = torch.ones(n1, dtype=dtype, device=device) / n1
    if marginal_b is None:
        marginal_b = torch.ones(n2, dtype=dtype, device=device) / n2
    if log_marginal_a is None:
        log_marginal_a = torch.log(marginal_a)
    if log_marginal_b is None:
        log_marginal_b = torch.log(marginal_b)

    f = torch.zeros(n1, dtype=dtype, device=device)
    g = torch.zeros(n2, dtype=dtype, device=device)

    if init_s is not None and init_s.shape == inter_c.shape:
        s = init_s.detach().to(device=device, dtype=dtype).clone()
    else:
        s = torch.ones((n1, n2), dtype=dtype, device=device) / (n1 * n2)

    gamma = torch.as_tensor(gamma_p, dtype=dtype, device=device)

    for _ in range(out_iter):
        s_hat = s - threshold_lambda
        a_hat = torch.sum(s_hat, dim=1)
        b_hat = torch.sum(s_hat, dim=0)
        left = _left_matvec(intra_c1_sq, a_hat).view(-1, 1)
        right = _row_vec_matmul(b_hat, intra_c2_sq).view(1, -1)
        L = left + right - 2.0 * _left_right_cost(intra_c1, s_hat, intra_c2)
        Q = inter_c + gw_weight * L

        for _ in range(in_iter):
            f = -gamma * torch.logsumexp(-(Q - g.view(1, -1)) / gamma, dim=1) + gamma * log_marginal_a
            g = -gamma * torch.logsumexp(-(Q - f.view(-1, 1)) / gamma, dim=0) + gamma * log_marginal_b

        exponent = (f.view(-1, 1) + g.view(1, -1) - Q) / gamma
        # Mild clipping prevents rare overflow in float32 without changing the
        # ordering of normal transport costs.
        exponent = torch.clamp(exponent, min=-80.0, max=40.0)
        s_new = torch.exp(exponent)
        s = 0.05 * s + 0.95 * s_new

    return s
