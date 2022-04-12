from typing import Optional, Union
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pickle as pkl

import numpy as np
import numpy.typing as npt
from scipy.special import digamma

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from bibim.hdp import gaussian as hdpgmm
from bibim.hdp.gaussian import HDPGMM
from bibim.data import MVVarSeqData

from .data import (HDFMultiVarSeqDataset,
                   collate_var_len_seq)
from .math import th_batch_logdotexp_2d as blogdotexp


CORPUS_LEVEL_PARAMS =  {'m', 'C', 'beta', 'nu', 'u', 'v', 'y', 'w2'}
_LOG_2 = torch.log(torch.as_tensor(2.))
_LOG_PI = torch.log(torch.as_tensor(torch.pi))
_LOG_2PI = _LOG_2 + _LOG_PI


@dataclass
class HDPGMM_GPU:
    hdpgmm: HDPGMM
    whiten_params: Optional[dict[str, npt.ArrayLike]] = None


def compute_ln_p_phi_x(
    data_batch: torch.Tensor,
    m: torch.Tensor,
    W_chol: torch.Tensor,
    W_logdet: torch.Tensor,
    beta: torch.Tensor,
    nu: torch.Tensor
) -> torch.Tensor:
    """
    """
    K = m.shape[0]
    M, max_len, D = data_batch.shape
    Eq_eta = torch.zeros((M, max_len, K), device=data_batch.device)
    _log2pi = _LOG_2PI.detach().clone().to(data_batch.device)

    # compute Eq_eta
    for k in range(K):
        Wx_k = (data_batch - m[k]) @ W_chol[k]
        Eq_eta[:, :, k] = (
            -.5 * (
                D * _log2pi
                + D * beta[k]**-1
                + nu[k] * (Wx_k**2).sum(-1)
                - W_logdet[k]
            )
        )
    return Eq_eta


def lnB(
    W: torch.Tensor,
    nu: float,
    logdet_W: Optional[float] = None
) -> float:
    """
    """
    d = W.shape[0]
    range_d = torch.arange(d).to(W.device)
    if logdet_W is None:
        sign, logdet_W = torch.linalg.slogdet(W)
        if sign <= 0:
            raise ValueError('[ERROR] log det of W must be positive!')

    return (
        -.5 * nu * logdet_W
        -.5 * nu * d * _LOG_2.to(W.device)
        -.25 * d * (d - 1) * _LOG_PI.to(W.device)
        -torch.special.gammaln(.5 * (nu - range_d)).sum()
    )


def compute_normalwishart_probs(
    m: torch.Tensor,
    C: torch.Tensor,
    nu: torch.Tensor,
    beta: torch.Tensor,
    W_chol: torch.Tensor,
    W_logdet: torch.Tensor,
    m0: torch.Tensor,
    W0_inv: torch.Tensor,
    nu0: float,
    beta0: float
) -> float:
    """
    """
    K, D = m.shape
    _log2pi = _LOG_2PI.detach().clone().to(m.device)

    # pre-compute some stuffs
    lnB0 = lnB(torch.linalg.inv(W0_inv), nu0)

    Eq_ln_p = 0.
    Eq_ln_q = 0.
    for k in range(K):
        # compute W_k and ln|W_k|
        W_k = torch.linalg.inv(C[k] * nu[k])
        logdet_W_k = torch.linalg.slogdet(W_k)[1]

        # compute H[q(precision_k)]
        H_k = (
            -lnB(W_k, nu[k], logdet_W_k)
            -(nu[k] - D - 1) * .5 * W_logdet[k]  # Eq[ln_prec_k]
            + nu[k] * D * .5
        )

        Eq_ln_p += (
            .5 * (
                D * (torch.log(torch.as_tensor(beta0)) - _log2pi)
                + W_logdet[k]
                - D * beta0 * beta[k]**-1
                - beta0 * nu[k] * torch.sum(((m[k] - m0) @ W_chol[k])**2)
            )
            + lnB0
            + (nu0 - D - 1) * .5 * W_logdet[k]
            - .5 * nu[k] * torch.trace(W0_inv @ W_k)
        )

        Eq_ln_q += (
            .5 * W_logdet[k]
            + .5 * D * (torch.log(torch.as_tensor(beta[k])) - _log2pi - 1.)
            - H_k
        )

    return Eq_ln_p - Eq_ln_q


def e_step(
    batch_idx: torch.LongTensor,
    data_batch: torch.Tensor,
    mask_batch: torch.BoolTensor,
    Eq_eta_batch: torch.Tensor,
    zeta: torch.Tensor,
    varphi: torch.Tensor,
    w_: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    w2: torch.Tensor,
    s1: float,
    s2: float,
    Eq_ln_beta: torch.Tensor,
    share_alpha0: bool = True,
    e_step_tol: float = 1e-4,
    n_max_iter: int = 100,
    eps: float = 1e-100
) -> float:
    """
    """
    M = data_batch.shape[0]
    T = a.shape[1] + 1
    Eq_ln_pi = torch.empty((M, T), device=Eq_eta_batch.device)

    cur_alpha0 = w2[0] / w2[1]
    ii = 0
    ln_lik = -torch.finfo().max
    converge = 1.
    old_ln_lik = -torch.finfo().max
    while (converge > 0. or converge > e_step_tol) and ii <= n_max_iter:

        a[batch_idx] = 1.
        if share_alpha0:
            b[batch_idx] = cur_alpha0
        else:
            b[batch_idx] = (w[batch_idx, 0] / w[batch_idx, 1])[:, None]

        varphi.copy_(
            Eq_ln_beta[None, None]
            + torch.bmm(zeta.permute(0, 2, 1), Eq_eta_batch)
        )
        varphi -= torch.logsumexp(varphi, dim=-1)[:, :, None]
        a[batch_idx] += zeta[:, :, :-1].sum(1)
        b[batch_idx] += (
            torch.flip(
                torch.cumsum(
                    torch.sum(torch.flip(zeta, dims=(2,)), dim=1),
                    dim=1
                ),
                dims=(1,)
            )[:, 1:]
        )

        ab_sum = torch.digamma(a[batch_idx] + b[batch_idx])
        Eq_ln_pi_hat = torch.digamma(a[batch_idx]) - ab_sum
        Eq_ln_1_min_pi_hat = torch.digamma(b[batch_idx]) - ab_sum
        Eq_ln_1_min_pi_hat_cumsum = torch.cumsum(Eq_ln_1_min_pi_hat, dim=1)
        Eq_ln_pi[:] = 0.
        Eq_ln_pi[:, :T-1] = Eq_ln_pi_hat
        Eq_ln_pi[:, 1:] += Eq_ln_1_min_pi_hat_cumsum

        zeta.copy_(
            Eq_ln_pi[:,None]
            + torch.bmm(Eq_eta_batch, torch.exp(varphi).permute(0, 2, 1))
        )
        zeta -= torch.logsumexp(zeta, dim=-1)[:, :, None]
        torch.exp(zeta, out=zeta)
        torch.clamp(zeta, min=eps, out=zeta)
        # zeta *= mask_batch[:, :, None]

        w[batch_idx, 0] = s1 + T - 1
        w[batch_idx, 1] = s2 - Eq_ln_1_min_pi_hat_cumsum[:, -1]

        Eq_ln_q_pi = (
            (a[batch_idx] - 1.) * Eq_ln_pi_hat
            + (b[batch_idx] - 1.) * Eq_ln_1_min_pi_hat
            - torch.special.gammaln(a[batch_idx]) - torch.special.gammaln(b[batch_idx])
            + torch.special.gammaln(a[batch_idx] + b[batch_idx])
        )

        if share_alpha0:
            alpha0 = cur_alpha0
        else:
            alpha0 = w[batch_idx, 0] / w[batch_idx, 1]

        Eq_ln_p_pi = (
            (alpha0 - 1.) * Eq_ln_1_min_pi_hat_cumsum[:, -1]
            - (T - 1.) * torch.log(alpha0)
        )

        ln_p_z = 0.
        Eq_ln_pz = 0.
        Eq_ln_pc = (torch.exp(varphi) @ Eq_ln_beta).sum()
        Eq_ln_qzqc = (
            (torch.exp(varphi) * varphi).sum()
            + torch.masked_select(zeta * zeta.log(), mask_batch[..., None]).sum()
        )
        ln_p_z += torch.masked_select(zeta * torch.bmm(Eq_eta_batch,
                                                       torch.exp(varphi).permute(0, 2, 1)),
                                      mask_batch[..., None]).sum()
        Eq_ln_pz += torch.bmm(zeta, Eq_ln_pi[:, :, None]).sum()

        # compute likelihood
        ln_lik = (
            ln_p_z
            + Eq_ln_pz + Eq_ln_pc - Eq_ln_qzqc
            + Eq_ln_p_pi.sum() - Eq_ln_q_pi.sum()
        )
        converge = (ln_lik - old_ln_lik) / abs(old_ln_lik)
        old_ln_lik = ln_lik

        # update counter
        ii += 1

    # update w tmp cumulator
    w_[0] = (T - 1.) * len(batch_idx)
    w_[1] = Eq_ln_1_min_pi_hat_cumsum[:, -1].sum(0)

    return ln_lik


def m_step(
    data_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    n_total_docs: int,
    m: torch.Tensor,
    C: torch.Tensor,
    nu: torch.Tensor,
    beta: torch.Tensor,
    zeta: torch.Tensor,
    varphi: torch.Tensor,
    w_: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    w2: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    y: torch.Tensor,
    Eq_ln_1_min_beta_hat: torch.Tensor,
    m0: torch.Tensor,
    W0_inv: torch.Tensor,
    nu0: float,
    beta0: float,
    s1: float,
    s2: float,
    g1: float,
    g2: float,
    noise_ratio: float,
    chunk_size: int = 128,
    eps: float = 1e-100
):
    """
    """
    M, max_len, D = data_batch.shape
    J = n_total_docs
    N_b, T, K = varphi.shape

    x_bar = torch.zeros((K, D), dtype=torch.float32, device=data_batch.device)
    S = torch.zeros((K, D, D), dtype=torch.float32, device=data_batch.device)
    N = torch.zeros((K,), dtype=torch.float32, device=data_batch.device)

    # for memory efficiency, we do the chunking here
    n_chunks = M // chunk_size + (M % chunk_size != 0)
    for i in range(n_chunks):
        slc = slice(i * chunk_size, (i+1) * chunk_size)
        zeta_chunk = zeta[slc]
        data_chunk = data_batch[slc]
        mask_chunk = mask_batch[slc].float()
        varphi_chunk = varphi[slc]
        r = zeta_chunk @ torch.clamp(torch.exp(varphi_chunk), min=eps)

        if noise_ratio > 0:
            norm = r.sum(-1)
            r /= torch.clamp(norm[:, :, None], min=eps)
            e = torch.rand(*r.shape, device=data_batch.device)
            e /= e.sum(-1)[:, :, None]
            e *= mask_chunk[:, :, None]
            r *= (1. - noise_ratio)
            r += e * noise_ratio
            r *= norm[:, :, None] * mask_chunk[:, :, None]

        N += r.sum((0, 1))
        for k in range(K):
            x_bar[k] += torch.bmm(r[:, :, k][:, None], data_chunk).sum(0)[0]
            rx = r[:, :, k][:, :, None]**.5 * data_chunk
            S[k] += torch.bmm(rx.permute(0, 2, 1), rx).sum(0)

    u_ = torch.exp(varphi[:, :, :-1]).sum((0, 1))
    v_ = torch.flip(
        torch.cumsum(
            torch.sum(torch.flip(torch.exp(varphi), dims=(2,)), dim=(0, 1)),
            dim=0
        ),
        dims=(0,)
    )[1:]

    # batch weight
    batch_w = float(J / M)

    # update alpha0
    w2[0] = s1 + w_[0] * batch_w
    w2[1] = s2 - w_[1] * batch_w

    # update gamma
    y[0] = g1 + K - 1
    y[1] = g2 - Eq_ln_1_min_beta_hat.sum()
    gamma = y[0] / y[1]

    # update beta
    u[:] = 1. + u_ * batch_w
    v[:] = gamma + v_ * batch_w

    # update Gaussian-Wishart
    N = N * batch_w
    for k in range(K):
        N_k = max(N[k], eps)
        nu[k] = nu0 + N_k
        beta[k] = beta0 + N_k

        x_bar_k = x_bar[k] * batch_w / N_k
        S_k = S[k] * batch_w / N_k - torch.outer(x_bar_k, x_bar_k)

        m[k] = beta[k]**-1 * (beta0 * m0 + N_k * x_bar_k)

        dev = (x_bar_k - m0)
        C[k] = (
            W0_inv
            + N_k * S_k
            + beta0 * N_k / (beta0 + N_k) * torch.outer(dev, dev)
        ) / nu[k]


def update_parameters(
    cur_iter: int,
    tau0: float,
    kappa: float,
    batch_size: int,
    params: dict[str, torch.Tensor],
    old_params: dict[str, torch.Tensor],
    corpus_level_params: set[str] = CORPUS_LEVEL_PARAMS
):
    """
    """
    if batch_size <= 0:
        rho = 1.
    else:
        rho = (tau0 + cur_iter)**-kappa

    for name in corpus_level_params:
        new_param = (1. - rho) * old_params[name] + rho * params[name]
        params[name].copy_(new_param)
        old_params[name].copy_(params[name])

    # re-compute auxiliary variables
    W = torch.linalg.inv(params['C'] * params['nu'][:, None, None])
    params['W_chol'] = torch.linalg.cholesky(W)
    params['W_logdet'] = torch.linalg.slogdet(W).logabsdet

    # to compute the full Eq[log|lambda_k|]
    K, D = params['m'].shape
    arange_d = torch.arange(D, device=params['m'].device)
    log2 = torch.log(torch.as_tensor([2], device=params['m'].device))
    for k in range(K):
        params['W_logdet'][k] += (
            torch.digamma((params['nu'][k] - arange_d) * .5).sum()
            + D * log2.item()
        )


def _init_params(
    max_components_corpus: int,
    max_components_documents: int,
    loader: DataLoader,
    m0: Optional[torch.Tensor] = None,
    W0: Optional[torch.Tensor] = None,
    nu0: Optional[torch.Tensor] = None,
    beta0: Optional[torch.Tensor] = None,
    s1: float = 1.,
    s2: float = 1.,
    g1: float = 1.,
    g2: float = 1.,
    device: str = 'cpu',
    full_uniform_init: bool = True,
    warm_start_with: Optional[HDPGMM_GPU] = None
) -> dict[str, Union[torch.Tensor,
                     float,
                     list[float]]]:
    """
    """
    J = len(loader.dataset)
    K = max_components_corpus
    T = max_components_documents
    N_, D = loader.dataset._hf['data'].shape
    W0_inv = np.linalg.inv(W0)


    if (warm_start_with is not None and
            isinstance(warm_start_with, HDPGMM_GPU)):

        # unpack for further monitoring down below
        model = warm_start_with
        mean_holdout_probs = model.training_monitors['mean_holdout_perplexity']
        train_lik = model.training_monitors['training_lowerbound']
        start_iter = len(model.training_monitors['training_lowerbound'])

        a = np.empty((J, T - 1))
        b = np.empty((J, T - 1))
        w = np.empty((J, 2), dtype=np.float64)
        w[:, 0] = s1
        w[:, 1] = s2
        w2 = np.empty((2,))

        u = np.empty((K - 1,))
        v = np.empty((K - 1,))
        y = np.empty((2,))

        m = np.empty((K, D))
        C = np.empty((K, D, D))
        beta = np.empty((K,))
        nu = np.empty((K,))

        for k, phi_ in enumerate(model.variational_params[0]):
            m[k] = phi_.mu0
            C[k] = np.linalg.inv(phi_.W) / phi_.nu
            beta[k] = phi_.lmbda
            nu[k] = phi_.nu

        u = model.variational_params[1].alphas
        v = model.variational_params[1].betas

        y[0] = model.variational_params[2].alpha
        y[1] = model.variational_params[2].beta
        if isinstance(model.variational_params[3], list):
            # per-document alpha0
            for j, w_ in enumerate(model.variational_params[3]):
                w[j, 0] = w_.alpha
                w[j, 1] = w_.beta
            w2[0] = g1
            w2[1] = g2
        else:
            # share alpha0 for all documents
            w2[0] = model.variational_params[3].alpha
            w2[1] = model.variational_params[3].beta

    else:
        # here we compute the parameters from data
        mean_holdout_probs = []
        train_lik = []
        start_iter = 0

        a = np.ones((J, T - 1), dtype=np.float64)
        b = np.full((J, T - 1), s1 / s2, dtype=np.float64)
        w = np.empty((J, 2), dtype=np.float64)
        w[:, 0] = s1
        w[:, 1] = s2
        w2 = np.array([s1, s2], dtype=np.float64)

        u = np.ones((K - 1,), dtype=np.float64)
        v = np.ones((K - 1,), dtype=np.float64)
        y = np.array([g1, g2], dtype=np.float64)

        m = np.zeros((K, D), dtype=np.float64)
        C = np.zeros((K, D, D), dtype=np.float64)

        x_bar = torch.zeros((K, D), device=device)
        S = torch.zeros((K, D, D), device=device)
        N = torch.zeros((K,), device=device)
        for mask_batch, data_batch, _ in loader:

            # "flatten" the multivariate vectors using the mask,
            # which is much easier form to work with this
            N_ = int(mask_batch.sum().item())  # the number of frames within this batch
            M = mask_batch.shape[0]  # batch size
            chunk = torch.empty((N_, data_batch.shape[-1]),
                                dtype=data_batch.dtype,
                                device=device)
            last = 0
            for j in range(M):
                n_j = int(mask_batch[j].sum().item())
                chunk[last:last + n_j] = data_batch[j, :n_j]
                last += n_j

            # now compute sufficient statistics
            if full_uniform_init:
                # populate uniform responsibility per frame/token
                r = torch.rand(N_, K).to(device)
                r /= r.sum(1)[:, None]

                # compute the stat and add to buffer
                N += r.sum(0)
                x_bar += r.T @ chunk
                for k in range(K):
                    S[k] += (r[:, k][:, None] * chunk).T @ chunk

            else:
                # populate uniform component selection per frame/token
                r = torch.randint(K, (N_,))
                for k in range(K):
                    x_k = chunk[r == k]
                    N[k] += (r == k).sum()
                    x_bar[k] += x_k.sum(0)
                    S[k] += x_k.T @ x_k

        # normalize stats
        x_bar /= N[:, None]
        S /= N[:, None, None]

        # set the params back to cpu/numpy side
        # TODO: this can be avoided by making the entire procedure
        #       into "torch" based program
        N = N.detach().cpu().numpy()
        x_bar = x_bar.detach().cpu().numpy()
        S = S.detach().cpu().numpy()

        # compute final initialization
        nu = nu0 + N
        beta = beta0 + N
        cov_reg = 1e-6 * np.eye(D)
        for k in range(K):
            # the model currently only works with float64
            x_bar_k = x_bar[k].astype(np.float64)
            S_k = S[k].astype(np.float64) - np.outer(x_bar_k, x_bar_k)

            # x_bar_ is not yet normalized (Nx_bar)
            m[k] = beta[k]**-1 * (beta0 * m0 + N[k] * x_bar_k)  # N_k * x_bar_k

            # same for S_ (it's NS) actually
            # this is terribly slow: scikit-learn DP-GMM circumvent it
            # through working with cholesky-decomposition of precision matrix
            # for almost everything (no inversion)
            dev = (x_bar_k - m0)
            C[k] = (
                W0_inv
                + N[k] * S_k  # N_k * S_k
                + beta0 * N[k] / (beta0 + N[k]) * np.outer(dev, dev)
                + cov_reg
            ) / nu[k]  # normalization as we compute "covariances" here

    # some pre-computations for computing Eq[eta] and Eq[a(eta)]
    W = np.linalg.inv(C * nu[:, None, None])
    W_chol = np.linalg.cholesky(W)
    W_logdet = np.linalg.slogdet(W)[1]

    # to compute the full Eq[log|lambda_k|]
    for k in range(K):
        W_logdet[k] += (
            digamma((nu[k] - np.arange(D)) * .5).sum()
            + D * _LOG_2
        )

    # prep the containor
    params = {}
    params['m'] = torch.as_tensor(m, dtype=torch.float32, device=device)
    params['C'] = torch.as_tensor(C, dtype=torch.float32, device=device)
    params['beta'] = torch.as_tensor(beta, dtype=torch.float32, device=device)
    params['nu'] = torch.as_tensor(nu, dtype=torch.float32, device=device)
    params['u'] = torch.as_tensor(u, dtype=torch.float32, device=device)
    params['v'] = torch.as_tensor(v, dtype=torch.float32, device=device)
    params['y'] = torch.as_tensor(y, dtype=torch.float32, device=device)
    params['a'] = torch.as_tensor(a, dtype=torch.float32, device=device)
    params['b'] = torch.as_tensor(b, dtype=torch.float32, device=device)
    params['w'] = torch.as_tensor(w, dtype=torch.float32, device=device)
    params['w2'] = torch.as_tensor(w2, dtype=torch.float32, device=device)
    params['w_cumul'] = torch.zeros((2,), dtype=torch.float32, device=device)
    params['W_chol'] = torch.as_tensor(W_chol, dtype=torch.float32, device=device)
    params['W_logdet'] = torch.as_tensor(W_logdet, dtype=torch.float32, device=device)
    params['m0'] = torch.as_tensor(m0, dtype=torch.float32, device=device)
    params['W0'] = torch.as_tensor(W0, dtype=torch.float32, device=device)
    params['W0_inv'] = torch.linalg.inv(params['W0'])
    params['nu0'] = nu0
    params['beta0'] = beta0
    params['s1'] = s1
    params['s2'] = s2
    params['g1'] = g1
    params['g2'] = g2
    params['mean_holdout_perplexity'] = mean_holdout_probs
    params['training_lowerbound'] = train_lik
    params['start_iter'] = start_iter
    if loader.dataset.whiten:
        params['whiten_params'] = loader.dataset._whitening_params

    return params


def init_params(
    max_components_corpus: int,
    max_components_documents: int,
    loader: DataLoader,
    m0: Optional[torch.Tensor] = None,
    W0: Optional[torch.Tensor] = None,
    nu0: Optional[torch.Tensor] = None,
    beta0: Optional[torch.Tensor] = None,
    s1: float = 1.,
    s2: float = 1.,
    g1: float = 1.,
    g2: float = 1.,
    device: str = 'cpu',
    n_W0_cluster: int = 128,
    cluster_frac: float = .01,
    full_uniform_init: bool = True,
    warm_start_with: Optional[HDPGMM_GPU] = None
) -> tuple[dict[str, Union[torch.Tensor,
                           float,
                           list[float]]],
           dict[str, torch.Tensor]]:
    """
    """
    # get some vars set
    K = max_components_corpus
    T = max_components_documents

    # build temporary MVVarSeqData from hdf file pointer
    mvvarseqdat = MVVarSeqData(loader.dataset._hf['indptr'][:],
                               loader.dataset._hf['data'],
                               loader.dataset._hf['ids'][:])

    # set / load / sample the hyper priors
    # TODO: this can be slow if the dataset get larger
    #       torch-gpu implementation could sped up this routine
    m0, W0, beta0, nu0 = hdpgmm.init_hyperprior(mvvarseqdat,
                                                m0, W0, nu0, beta0,
                                                n_W0_cluster, cluster_frac,
                                                warm_start_with=warm_start_with)

    # get initialization
    params = _init_params(K, T, loader,
                          m0=m0, W0=W0, nu0=nu0, beta0=beta0,
                          s1=s1, s2=s2, g1=g1, g2=g2,
                          full_uniform_init=full_uniform_init,
                          warm_start_with=warm_start_with,
                          device=device)

    # prep the "old params" to compute the improvement
    old_params = {name: params[name].clone().detach()
                  for name in CORPUS_LEVEL_PARAMS}

    return params, old_params


def package_model(
    max_components_corpus: int,
    max_components_document: int,
    params: dict[str, torch.Tensor],
    mean_holdout_probs: list[float],
    train_lik: list[float],
    share_alpha0: bool = False
) -> dict[str, HDPGMM_GPU]:
    """
    """
    pkg = HDPGMM_GPU(
        hdpgmm = hdpgmm.package_model(
            max_components_corpus,
            max_components_document,
            params['m'].detach().cpu().numpy(),
            params['C'].detach().cpu().numpy(),
            params['nu'].detach().cpu().numpy(),
            params['beta'].detach().cpu().numpy(),
            params['w'].detach().cpu().numpy(),
            params['w2'].detach().cpu().numpy(),
            params['u'].detach().cpu().numpy(),
            params['v'].detach().cpu().numpy(),
            params['y'].detach().cpu().numpy(),
            params['m0'].detach().cpu().numpy(),
            params['W0'].detach().cpu().numpy(),
            params['beta0'],
            params['nu0'],
            params['s1'], params['s2'], params['g1'], params['g2'],
            mean_holdout_probs, train_lik, share_alpha0
        ),
        whiten_params = params.get('whiten_params')
    )
    return pkg


# def infer_document():


def variational_inference(
    h5_fn: str,
    max_components_corpus: int,
    max_components_document: int,
    n_epochs: int,
    batch_size: int = 512,
    kappa: float = .5,
    tau0: float = 1.,
    m0: Optional[npt.NDArray[np.float64]] = None,
    W0: Optional[npt.NDArray[np.float64]] = None,
    nu0: Optional[float] = None,
    beta0: float = 2.,
    s1: float = 1.,
    s2: float = 1.,
    g1: float = 1.,
    g2: float = 1.,
    n_max_inner_iter: int = 200,
    e_step_tol: float = 1e-4,
    base_noise_ratio: float = 1e-4,
    full_uniform_init: bool = True,
    share_alpha0: bool = True,
    whiten: bool = False,
    data_parallel_num_workers: int = 0,
    n_W0_cluster: int = 128,
    cluster_frac: float = .01,
    warm_start_with: Optional[HDPGMM_GPU] = None,
    max_len: Optional[int] = None,
    save_every: Optional[int] = None,
    out_path: str = './',
    prefix: str = 'hdp_gmm',
    eps: float = torch.finfo().eps,
    device: str = 'cpu',
    verbose: bool = False
):
    """
    """
    ###################
    # Loading Dataset
    ###################
    if warm_start_with and warm_start_with.whiten_params:
        dataset = HDFMultiVarSeqDataset(h5_fn)
        dataset._whitening_params = warm_start_with.whiten_params
    else:
        dataset = HDFMultiVarSeqDataset(h5_fn, whiten=whiten)
    loader = DataLoader(
        dataset,
        num_workers=data_parallel_num_workers,
        collate_fn=partial(collate_var_len_seq, max_len=max_len),
        batch_size=batch_size,
        shuffle=True
    )

    #######################
    # Initializing Params
    #######################
    params, old_params = init_params(
        max_components_corpus, max_components_document, loader,
        m0, W0, nu0, beta0, s1, s2, g1, g2,
        device, n_W0_cluster, cluster_frac,
        full_uniform_init=full_uniform_init,
        warm_start_with=warm_start_with
    )

    #######################
    # Update loop!
    #######################
    trn_liks = params['training_lowerbound']
    mean_holdout_probs = params['mean_holdout_perplexity']
    it = params['start_iter']
    try:
        with tqdm(total=n_epochs, ncols=80, disable=not verbose) as prog:
            for _ in range(n_epochs):
                for mask_batch, data_batch, batch_idx in loader:

                    # send tensors to the target device
                    mask_batch = mask_batch.to(device)
                    data_batch = data_batch.to(device)
                    batch_idx = batch_idx.to(device)

                    with torch.no_grad():

                        # init some variables (including accumulators?)
                        varphi = torch.zeros((data_batch.shape[0],
                                              max_components_document,
                                              max_components_corpus),
                                             device=data_batch.device)
                        zeta = torch.rand(data_batch.shape[0],
                                          data_batch.shape[1],
                                          max_components_document).to(device)
                        zeta /= zeta.sum(2)[:, :, None]
                        zeta *= mask_batch[:, :, None]

                        # compute Eq_beta and probability
                        uv_sum = torch.digamma(params['u'] + params['v'])
                        Eq_ln_beta_hat = torch.digamma(params['u']) - uv_sum
                        Eq_ln_1_min_beta_hat = torch.digamma(params['v']) - uv_sum
                        Eq_ln_beta = torch.zeros((max_components_corpus,),
                                                 device=data_batch.device)
                        Eq_ln_beta[:max_components_corpus-1] = Eq_ln_beta_hat
                        Eq_ln_beta[1:] += torch.cumsum(Eq_ln_1_min_beta_hat, dim=0)

                        gamma = params['y'][0] / params['y'][1]
                        Eq_ln_p_beta = (
                            (gamma - 1.) * Eq_ln_1_min_beta_hat.sum()
                            - (max_components_corpus - 1.) * torch.log(gamma)
                        )
                        Eq_ln_q_beta = (
                            (params['u'] - 1.) * Eq_ln_beta_hat
                            + (params['v'] - 1) * Eq_ln_1_min_beta_hat
                            - (torch.special.gammaln(params['u'])
                               + torch.special.gammaln(params['v'])
                               - torch.special.gammaln(params['u'] + params['v']))
                        ).sum()

                        # COMPUTE Eq[eta]
                        Eq_eta = compute_ln_p_phi_x(
                            data_batch,
                            params['m'],
                            params['W_chol'],
                            params['W_logdet'],
                            params['beta'],
                            params['nu']
                        )
                        # DO E-STEP AND COMPUTE BATCH LIKELIHOOD
                        ln_lik = e_step(
                            batch_idx, data_batch, mask_batch,
                            Eq_eta, zeta, varphi, params['w_cumul'],
                            params['a'], params['b'], params['w'], params['w2'],
                            s1=params['s1'], s2=params['s2'],
                            Eq_ln_beta = Eq_ln_beta,
                            share_alpha0 = share_alpha0,
                            e_step_tol = e_step_tol,
                            n_max_iter = n_max_inner_iter,
                            eps=eps
                        )

                        # COMPUTE LOWERBOUNDS
                        ln_lik *= len(dataset) / data_batch.shape[0]  # est. for the total lik
                        nw_prob = compute_normalwishart_probs(
                            params['m'], params['C'],
                            params['nu'], params['beta'],
                            params['W_chol'], params['W_logdet'],
                            params['m0'], params['W0_inv'],
                            params['nu0'], params['beta0']
                        )
                        total_ln_lik_est = (
                            ln_lik + Eq_ln_p_beta - Eq_ln_q_beta + nw_prob
                        )
                        trn_liks.append(total_ln_lik_est.item())

                        # DO M-STEP
                        noise_ratio = base_noise_ratio * 1000. / (it + 1000.)
                        m_step(
                            data_batch, mask_batch, len(dataset),
                            params['m'], params['C'],
                            params['nu'], params['beta'],
                            zeta, varphi, params['w_cumul'],
                            params['a'], params['b'], params['w'], params['w2'],
                            params['u'], params['v'], params['y'],
                            Eq_ln_1_min_beta_hat,
                            params['m0'], params['W0_inv'],
                            params['nu0'], params['beta0'],
                            params['s1'], params['s2'],
                            params['g1'], params['g2'],
                            noise_ratio = noise_ratio,
                            eps = eps
                        )

                        # UPDATE NEW PARAMETERS
                        update_parameters(
                            cur_iter=it,
                            tau0=tau0,
                            kappa=kappa,
                            batch_size=batch_size,
                            params=params,
                            old_params=old_params
                        )

                        # free up some big variables to save mem
                        del Eq_eta
                        del zeta
                        del varphi
                        torch.cuda.empty_cache()

                        if save_every is not None and it % save_every == 0:
                            ret = package_model(
                                max_components_corpus,
                                max_components_document,
                                params,
                                mean_holdout_probs,
                                trn_liks,
                                share_alpha0
                            )
                            path = Path(out_path) / f'{prefix}_it{it:d}.pkl'
                            with path.open('wb') as fp:
                                pkl.dump(ret, fp)

                    it += 1
                prog.update()

    except KeyboardInterrupt as ke:
        print('User interrupted the training! finishing the fitting...')
    except Exception as e:
        raise e

    # wrap the result and return
    ret = package_model(
        max_components_corpus,
        max_components_document,
        params,
        mean_holdout_probs,
        trn_liks,
        share_alpha0
    )
    return ret
