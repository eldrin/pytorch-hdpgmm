from typing import Optional, Union
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pickle as pkl

import numpy as np
import numpy.typing as npt
from scipy.special import digamma

import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from bibim.hdp import gaussian as hdpgmm
from bibim.hdp.gaussian import HDPGMM
from bibim.data import MVVarSeqData

from .data import (HDFMultiVarSeqDataset,
                   collate_var_len_seq)
from .math import th_masked_logsumexp as masked_logsumexp


CORPUS_LEVEL_PARAMS =  {'m', 'C', 'beta', 'nu', 'u', 'v', 'y', 'w2'}
_LOG_2 = torch.log(torch.as_tensor(2.))
_LOG_PI = torch.log(torch.as_tensor(torch.pi))
_LOG_2PI = _LOG_2 + _LOG_PI


@dataclass
class HDPGMM_GPU:
    hdpgmm: HDPGMM
    whiten_params: Optional[dict[str, npt.ArrayLike]] = None


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


def compute_ln_p_phi_x(
    data_batch: torch.Tensor,
    mask_batch: torch.BoolTensor,
    m: torch.Tensor,
    W: torch.Tensor,
    W_chol: torch.Tensor,
    W_logdet: torch.Tensor,
    W_isposdef: torch.BoolTensor,
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
        if W_isposdef[k]:
            Wx_k = (data_batch - m[k]) @ W_chol[k]
            Wx_k = (Wx_k**2).sum(-1)
        else:
            dif = (data_batch - m[k])
            Wx_k = ((dif @ W[k]) * dif).sum(-1)
        Wx_k *= mask_batch

        Eq_eta[:, :, k] = (
            -.5 * (
                D * _log2pi
                + D * beta[k]**-1
                + nu[k] * Wx_k
                - W_logdet[k]
            )
        )
    Eq_eta *= mask_batch[..., None]
    return Eq_eta


def compute_corpus_stick(
    max_components_corpus: int,
    params: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    """
    K = max_components_corpus
    corpus_stick = {}

    uv_sum = torch.digamma(params['u'] + params['v'])
    corpus_stick['Eq_ln_beta_hat'] = torch.digamma(params['u']) - uv_sum
    corpus_stick['Eq_ln_1_min_beta_hat'] = torch.digamma(params['v']) - uv_sum
    corpus_stick['Eq_ln_beta'] = torch.zeros((K,), device=params['u'].device)
    corpus_stick['Eq_ln_beta'][:K-1] = corpus_stick['Eq_ln_beta_hat']
    corpus_stick['Eq_ln_beta'][1:] += torch.cumsum(
        corpus_stick['Eq_ln_1_min_beta_hat'],
        dim=0
    )

    gamma = params['y'][0] / params['y'][1]
    corpus_stick['Eq_ln_p_beta'] = (
        (gamma - 1.) * corpus_stick['Eq_ln_1_min_beta_hat'].sum()
        - (K - 1.) * torch.log(gamma)
    )
    corpus_stick['Eq_ln_q_beta'] = (
        (params['u'] - 1.) * corpus_stick['Eq_ln_beta_hat']
        + (params['v'] - 1) * corpus_stick['Eq_ln_1_min_beta_hat']
        - (torch.special.gammaln(params['u'])
           + torch.special.gammaln(params['v'])
           - torch.special.gammaln(params['u'] + params['v']))
    ).sum()

    return corpus_stick


def compute_document_stick(
    max_components_document: int,
    a: torch.Tensor,
    b: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    """
    M = a.shape[0]  # batch_size
    T = max_components_document
    doc_stick = {}
    doc_stick['Eq_ln_pi'] = torch.empty((M, T), device=a.device)

    ab_sum = torch.digamma(a + b)
    doc_stick['Eq_ln_pi_hat'] = torch.digamma(a) - ab_sum
    doc_stick['Eq_ln_1_min_pi_hat'] = torch.digamma(b) - ab_sum
    doc_stick['Eq_ln_1_min_pi_hat_cumsum'] = (
        torch.cumsum(doc_stick['Eq_ln_1_min_pi_hat'], dim=1)
    )
    doc_stick['Eq_ln_pi'][:] = 0.
    doc_stick['Eq_ln_pi'][:, :T-1] = doc_stick['Eq_ln_pi_hat']
    doc_stick['Eq_ln_pi'][:, 1:] += doc_stick['Eq_ln_1_min_pi_hat_cumsum']

    return doc_stick


def compute_normalwishart_probs(
    m: torch.Tensor,
    W: torch.Tensor,
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
        W_k = W[k]
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
    params: dict[str, torch.Tensor],
    temp_vars: dict[str, torch.Tensor],
    corpus_stick: dict[str, torch.Tensor],
    share_alpha0: bool = True,
    e_step_tol: float = 1e-4,
    n_max_iter: int = 100,
    noise_ratio: float = 1e-4,
    chunk_size: int = 128,
    return_responsibility: bool = False,
    eps: float = torch.finfo().eps
) -> tuple[float,                    # mini-batch likelihood
           dict[str, torch.Tensor],  # document stick latent vars inference
           Optional[torch.Tensor]]:  # responsibilities (optional)
    """
    """
    M = data_batch.shape[0]
    T = params['a'].shape[1] + 1
    K = params['u'].shape[0] + 1
    if return_responsibility:
        resp_batch = torch.empty((M, K),
                                 dtype=params['m'].dtype,
                                 device=params['m'].device)
    else:
        resp_batch = None

    cur_alpha0 = params['w2'][0] / params['w2'][1]
    ii = 0
    ln_lik = -torch.finfo().max
    converge = 1.
    old_ln_lik = -torch.finfo().max
    while (converge > 0. or converge > e_step_tol) and ii <= n_max_iter:

        params['a'][batch_idx] = 1.
        if share_alpha0:
            params['b'][batch_idx] = cur_alpha0
        else:
            params['b'][batch_idx] = (
                params['w'][batch_idx, 0] / params['w'][batch_idx, 1]
            )[:, None]

        temp_vars['varphi'].copy_(
            corpus_stick['Eq_ln_beta'][None, None]
            + torch.bmm(temp_vars['zeta'].permute(0, 2, 1), Eq_eta_batch)
        )
        temp_vars['varphi'] -= torch.logsumexp(temp_vars['varphi'], dim=-1)[:, :, None]
        params['a'][batch_idx] += temp_vars['zeta'][:, :, :-1].sum(1)
        params['b'][batch_idx] += (
            torch.flip(
                torch.cumsum(
                    torch.sum(torch.flip(temp_vars['zeta'], dims=(2,)), dim=1),
                    dim=1
                ),
                dims=(1,)
            )[:, 1:]
        )

        doc_stick = compute_document_stick(
            T,
            params['a'][batch_idx],
            params['b'][batch_idx]
        )

        temp_vars['zeta'].copy_(
            doc_stick['Eq_ln_pi'][:,None]
            + torch.bmm(Eq_eta_batch,
                        torch.exp(temp_vars['varphi']).permute(0, 2, 1))
        )
        temp_vars['zeta'] -= torch.logsumexp(temp_vars['zeta'], dim=-1)[:, :, None]
        torch.exp(temp_vars['zeta'], out=temp_vars['zeta'])
        torch.clamp(temp_vars['zeta'], min=eps, out=temp_vars['zeta'])
        temp_vars['zeta'] *= mask_batch[:, :, None]

        params['w'][batch_idx, 0] = params['s1'] + T - 1
        params['w'][batch_idx, 1] = params['s2'] - doc_stick['Eq_ln_1_min_pi_hat_cumsum'][:, -1]

        Eq_ln_q_pi = (
            (params['a'][batch_idx] - 1.) * doc_stick['Eq_ln_pi_hat']
            + (params['b'][batch_idx] - 1.) * doc_stick['Eq_ln_1_min_pi_hat']
            - torch.special.gammaln(params['a'][batch_idx])
            - torch.special.gammaln(params['b'][batch_idx])
            + torch.special.gammaln(params['a'][batch_idx] + params['b'][batch_idx])
        )

        if share_alpha0:
            alpha0 = cur_alpha0
        else:
            alpha0 = params['w'][batch_idx, 0] / params['w'][batch_idx, 1]

        Eq_ln_p_pi = (
            (alpha0 - 1.) * doc_stick['Eq_ln_1_min_pi_hat_cumsum'][:, -1]
            - (T - 1.) * torch.log(alpha0)
        )

        exp_varphi = torch.exp(temp_vars['varphi'])
        # ln_p_z = 0.
        # Eq_ln_pz = 0.
        Eq_ln_pc = (exp_varphi @ corpus_stick['Eq_ln_beta']).sum()
        Eq_ln_qzqc = (
            (exp_varphi * temp_vars['varphi']).sum()
            + torch.masked_select(temp_vars['zeta'] * temp_vars['zeta'].log(),
                                  mask_batch[..., None]).sum()
        )
        ln_p_z = torch.masked_select(
            temp_vars['zeta'] * torch.bmm(Eq_eta_batch,
                                          exp_varphi.permute(0, 2, 1)),
            mask_batch[..., None]
        ).sum()
        Eq_ln_pz = torch.masked_select(
            torch.bmm(temp_vars['zeta'], doc_stick['Eq_ln_pi'][:, :, None]),
            mask_batch[..., None]
        ).sum()

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

    # update necessary statistics cumulators
    # -- update w tmp cumulator
    params['ss']['w_'][0] += (T - 1.) * len(batch_idx)
    params['ss']['w_'][1] += doc_stick['Eq_ln_1_min_pi_hat_cumsum'][:, -1].sum(0)

    # -- update N / x_bar / S
    # for memory efficiency, we do the chunking here
    n_chunks = M // chunk_size + (M % chunk_size != 0)
    for i in range(n_chunks):
        # chunk varibales
        slc = slice(i * chunk_size, (i+1) * chunk_size)
        zeta_chunk = temp_vars['zeta'][slc]
        # varphi_chunk = temp_vars['varphi'][slc]
        exp_varphi_chunk = exp_varphi[slc]
        data_chunk = data_batch[slc]
        mask_chunk = mask_batch[slc].float()

        # compute responsibility
        r = torch.bmm(zeta_chunk, torch.clamp(exp_varphi_chunk, min=eps))
        r *= mask_chunk[:, :, None]
        if return_responsibility:
            resp_batch[slc] = r.sum(1) / mask_chunk.sum(1).float()[:, None]

        # splash noise if needed
        if noise_ratio > 0:
            norm = r.sum(-1)
            r /= torch.clamp(norm[:, :, None], min=eps)
            e = torch.rand(*r.shape, device=data_batch.device)
            e /= e.sum(-1)[:, :, None]
            e *= mask_chunk[:, :, None]
            r *= (1. - noise_ratio)
            r += e * noise_ratio
            r *= norm[:, :, None] * mask_chunk[:, :, None]

        params['ss']['N'] += r.sum((0, 1))
        for k in range(K):
            params['ss']['x_bar'][k] += torch.bmm(r[:, :, k][:, None],
                                                  data_chunk).sum(0)[0]
            rx = r[:, :, k][:, :, None]**.5 * data_chunk
            params['ss']['S'][k] += torch.bmm(rx.permute(0, 2, 1), rx).sum(0)

    params['ss']['u_'] += exp_varphi[:, :, :-1].sum((0, 1))
    params['ss']['v_'] += torch.flip(
        torch.cumsum(
            torch.sum(torch.flip(exp_varphi, dims=(2,)), dim=(0, 1)),
            dim=0
        ),
        dims=(0,)
    )[1:]

    return ln_lik, doc_stick, resp_batch


def m_step(
    batch_size: int,
    n_total_docs: int,
    params: dict[str, torch.Tensor],
    corpus_stick: dict[str, torch.Tensor],
    cov_reg_weight: float = 1e-6,
    eps: float = torch.finfo().eps
):
    """
    """
    J = n_total_docs
    M = batch_size
    K, D = params['m'].shape

    # batch weight
    batch_w = float(J / M)

    # update alpha0
    params['w2'][0] = params['s1'] + params['ss']['w_'][0] * batch_w
    params['w2'][1] = params['s2'] - params['ss']['w_'][1] * batch_w

    # update gamma
    params['y'][0] = params['g1'] + K - 1
    params['y'][1] = params['g2'] - corpus_stick['Eq_ln_1_min_beta_hat'].sum()
    gamma = params['y'][0] / params['y'][1]

    # update beta
    params['u'][:] = 1. + params['ss']['u_'] * batch_w
    params['v'][:] = gamma + params['ss']['v_'] * batch_w

    # update Gaussian-Wishart
    N = params['ss']['N'] * batch_w
    cov_reg = cov_reg_weight * torch.eye(D, device=params['m'].device)
    for k in range(K):
        N_k = max(N[k], eps)
        params['nu'][k] = params['nu0'] + N_k
        params['beta'][k] = params['beta0'] + N_k

        x_bar_k = params['ss']['x_bar'][k] * batch_w / N_k
        xx_bar_k = torch.outer(x_bar_k, x_bar_k)
        S_k = params['ss']['S'][k] * batch_w / N_k - xx_bar_k

        params['m'][k] = (
            params['beta'][k]**-1
            * (params['beta0'] * params['m0'] + N_k * x_bar_k)
        )

        dev = (x_bar_k - params['m0'])
        dev2 = torch.outer(dev, dev)
        params['C'][k] = (
            params['W0_inv']
            + N_k * S_k
            + params['beta0'] * N_k / (params['beta0'] + N_k) * dev2
            + cov_reg
        ) / params['nu'][k]
        # params['C'][k] = (
        #     params['W0_inv']
        #     + N_k * S_k
        #     + params['beta0'] * N_k / (params['beta0'] + N_k) * dev2
        #     + cov_reg
        # )


def update_parameters(
    cur_iter: int,
    tau0: float,
    kappa: float,
    batch_size: int,
    params: dict[str, torch.Tensor],
    old_params: dict[str, torch.Tensor],
    batch_update: bool = False,
    corpus_level_params: set[str] = CORPUS_LEVEL_PARAMS,
    eps: float = 1e-6  # for regularizing precision matrix
):
    """
    """
    K, D = params['m'].shape

    if batch_size <= 0 or batch_update:
        rho = 1.
    else:
        rho = (tau0 + cur_iter)**-kappa

    for name in corpus_level_params:
        new_param = (1. - rho) * old_params[name] + rho * params[name]
        params[name].copy_(new_param)
        old_params[name].copy_(params[name])

    # re-compute auxiliary variables
    W = torch.linalg.inv(params['C'] * params['nu'][:, None, None])
    # W = torch.linalg.inv(params['C'])

    L, info = torch.linalg.cholesky_ex(W)
    assert info.sum() == 0
    params['W'].copy_(W)
    params['W_isposdef'].copy_(info == 0)
    params['W_chol'].copy_(L)
    params['W_logdet'].copy_(torch.linalg.slogdet(W).logabsdet)

    # to compute the full Eq[log|lambda_k|]
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
        model = warm_start_with.hdpgmm
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
        W = np.empty_like(C)
        beta = np.empty((K,))
        nu = np.empty((K,))

        for k, phi_ in enumerate(model.variational_params[0]):
            m[k] = phi_.mu0
            C[k] = np.linalg.inv(phi_.W) / phi_.nu
            # C[k] = np.linalg.inv(phi_.W)
            W[k] = phi_.W
            beta[k] = phi_.lmbda
            nu[k] = phi_.nu

        u = model.variational_params[1].alphas
        v = model.variational_params[1].betas

        y[0] = model.variational_params[2].alpha
        y[1] = model.variational_params[2].beta
        if isinstance(model.variational_params[3], list):
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
            # )

    # some pre-computations for computing Eq[eta] and Eq[a(eta)]
    W = np.linalg.inv(C * nu[:, None, None])
    # W = np.linalg.inv(C)
    W_logdet = np.linalg.slogdet(W)[1]

    # to compute the full Eq[log|lambda_k|]
    for k in range(K):
        W_logdet[k] += (
            digamma((nu[k] - np.arange(D)) * .5).sum()
            + D * _LOG_2
        )

    # prep the containor
    params = {}

    # init main parameters
    params['m'] = torch.as_tensor(m, dtype=torch.float32, device=device)
    params['C'] = torch.as_tensor(C, dtype=torch.float32, device=device)
    # params['C'] = torch.as_tensor(C / nu[:, None, None],
    #                               dtype=torch.float32, device=device)
    params['W'] = torch.as_tensor(W, dtype=torch.float32, device=device)
    params['beta'] = torch.as_tensor(beta, dtype=torch.float32, device=device)
    params['nu'] = torch.as_tensor(nu, dtype=torch.float32, device=device)
    params['u'] = torch.as_tensor(u, dtype=torch.float32, device=device)
    params['v'] = torch.as_tensor(v, dtype=torch.float32, device=device)
    params['y'] = torch.as_tensor(y, dtype=torch.float32, device=device)
    params['a'] = torch.as_tensor(a, dtype=torch.float32, device=device)
    params['b'] = torch.as_tensor(b, dtype=torch.float32, device=device)
    params['w'] = torch.as_tensor(w, dtype=torch.float32, device=device)
    params['w2'] = torch.as_tensor(w2, dtype=torch.float32, device=device)
    L, info = torch.linalg.cholesky_ex(params['W'])
    params['W_chol'] = torch.as_tensor(L, dtype=torch.float32, device=device)
    params['W_isposdef'] = info == 0
    params['W_logdet'] = torch.as_tensor(W_logdet, dtype=torch.float32, device=device)

    # copying (wrapping to torch Tensors) hyper-priors
    params['m0'] = torch.as_tensor(m0, dtype=torch.float32, device=device)
    params['W0'] = torch.as_tensor(W0, dtype=torch.float32, device=device)
    params['W0_inv'] = torch.linalg.inv(params['W0'])
    params['nu0'] = nu0
    params['beta0'] = beta0
    params['s1'] = s1
    params['s2'] = s2
    params['g1'] = g1
    params['g2'] = g2

    # initializing buffers for the sufficient statistics
    params['ss'] = {}
    params['ss']['x_bar'] = torch.zeros((K, D), dtype=torch.float32, device=device)
    params['ss']['S'] = torch.zeros((K, D, D), dtype=torch.float32, device=device)
    params['ss']['N'] = torch.zeros((K,), dtype=torch.float32, device=device)
    params['ss']['w_'] = torch.zeros((2,), dtype=torch.float32, device=device)
    params['ss']['u_'] = torch.zeros((K - 1,), dtype=torch.float32, device=device)
    params['ss']['v_'] = torch.zeros((K - 1,), dtype=torch.float32, device=device)

    # wrapping the model fits and others states
    params['mean_holdout_perplexity'] = mean_holdout_probs
    params['training_lowerbound'] = train_lik
    params['start_iter'] = start_iter
    if loader.dataset.whiten:
        params['whiten_params'] = loader.dataset._whitening_params

    return params


def _init_hyperpriors(
    loader: DataLoader,
    beta0: float = 2.,
    nu0_offset: float = 2.,
    device: str = 'cpu',
    verbose: bool = False
) -> tuple[npt.ArrayLike,
           npt.ArrayLike,
           npt.ArrayLike,
           npt.ArrayLike]:
    """
    """
    x_bar = None
    S = None
    N = 0
    with tqdm(total=len(loader), ncols=80, disable=not verbose) as prog:
        for mask_batch, data_batch, batch_idx in loader:

            # send tensors to the target device
            mask_batch = mask_batch.to(device)
            data_batch = data_batch.to(device)
            batch_idx = batch_idx.to(device)

            if x_bar is None:
                D = data_batch.shape[-1]
                x_bar = torch.zeros((D,), dtype=data_batch.dtype, device=device)
                S = torch.zeros((D, D), dtype=data_batch.dtype, device=device)

            # compute accumulators
            data_flat = data_batch.view(-1, data_batch.shape[-1])
            x_bar += data_flat.sum(0)
            S += data_flat.T @ data_flat
            N += data_flat.shape[0]

            prog.update()

    m0 = x_bar / N
    W0 = torch.linalg.inv((S / N) - torch.outer(m0, m0))

    m0 = m0.detach().cpu().numpy()
    W0 = W0.detach().cpu().numpy()
    nu0 = D + nu0_offset
    return m0, W0, nu0, beta0


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

    if warm_start_with:
        m0, W0, nu0, beta0 = (
            warm_start_with.hdpgmm.hyper_params[0].mu0,
            warm_start_with.hdpgmm.hyper_params[0].W,
            warm_start_with.hdpgmm.hyper_params[0].nu,
            warm_start_with.hdpgmm.hyper_params[0].lmbda
        )
    else:
        if any([prm is None for prm in [m0, W0, nu0, beta0]]):
            m0, W0, nu0, beta0 = _init_hyperpriors(
                loader, device=device
            )

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


def init_temp_vars(
    max_components_corpus: int,
    max_components_document: int,
    data_batch: torch.Tensor,
    mask_batch: torch.BoolTensor
) -> dict[str, torch.Tensor]:
    """
    initialize & allocate the temporary variables
        mostly including latent variables (i.e., zeta / varphi)
    """
    M, max_len, D = data_batch.shape
    K = max_components_corpus
    T = max_components_document
    device = data_batch.device

    tmp_vars = {}
    tmp_vars['varphi'] = torch.zeros((M, T, K),
                                     dtype=data_batch.dtype,
                                     device=device)
    tmp_vars['zeta'] = torch.rand(M, max_len, T).to(device)
    tmp_vars['zeta'] /= tmp_vars['zeta'].sum(-1)[:, :, None]
    tmp_vars['zeta'] *= mask_batch[:, :, None]
    return tmp_vars


def reinit_ss(
    params: dict[str, torch.Tensor]
) -> None:
    """
    """
    for k in params['ss'].keys():
        params['ss'][k][:] = 0.


def package_model(
    max_components_corpus: int,
    max_components_document: int,
    params: dict[str, torch.Tensor],
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
            # (params['C'] / params['nu'][:, None, None]).detach().cpu().numpy(),
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
            params['mean_holdout_perplexity'],
            params['training_lowerbound'],
            share_alpha0
        ),
        whiten_params = params.get('whiten_params')
    )
    return pkg


def infer_documents(
    dataset: HDFMultiVarSeqDataset,
    model: HDPGMM_GPU,
    n_max_inner_iter: int = 100,
    e_step_tol: float = 1e-4,
    batch_size: int = 512,
    device: str = 'cpu',
    max_len: int = torch.iinfo(torch.int32).max,
    eps: float = torch.finfo().eps
) -> dict[str, torch.Tensor]:
    """
    """
    K = model.hdpgmm.max_components_corpus
    T = model.hdpgmm.max_components_document
    if isinstance(model.hdpgmm.variational_params[3], list):
        share_alpha0 = False
    else:
        share_alpha0 = True

    # setup data loader
    loader = DataLoader(
        dataset,
        num_workers=0,
        collate_fn=partial(collate_var_len_seq, max_len=max_len),
        batch_size=batch_size,
        shuffle=False
    )

    # unpack model
    params = init_params(
        K, T, loader,
        warm_start_with=model,
        device=device
    )[0]

    # infer documents
    n_samples = len(loader.dataset)
    ln_lik_ = torch.empty((n_samples, K), device=device)
    ln_prior_ = torch.empty_like(ln_lik_)
    Eq_ln_pi_ = torch.empty((n_samples, T), device=device)
    for mask_batch, data_batch, batch_idx in loader:

        # send tensors to the target device
        mask_batch = mask_batch.to(device)
        data_batch = data_batch.to(device)
        batch_idx = batch_idx.to(device)

        with torch.no_grad():

            # init some variables
            # and re-init including accumulators for sufficient statistics
            temp_vars = init_temp_vars(K, T, data_batch, mask_batch)
            reinit_ss(params)

            # compute Eq_beta and probability
            corpus_stick = compute_corpus_stick(K, params)

            # COMPUTE Eq[eta]
            Eq_eta = compute_ln_p_phi_x(
                data_batch,
                mask_batch,
                params['m'],
                params['W'],
                params['W_chol'],
                params['W_logdet'],
                params['W_isposdef'],
                params['beta'],
                params['nu']
            )

            # DO E-STEP AND COMPUTE BATCH LIKELIHOOD
            _, doc_stick_batch, resp_batch = e_step(
                batch_idx, data_batch, mask_batch,
                Eq_eta, params, temp_vars, corpus_stick,
                share_alpha0 = share_alpha0,
                e_step_tol = e_step_tol,
                n_max_iter = n_max_inner_iter,
                noise_ratio = 0.,
                return_responsibility = True,
                eps = eps
            )

            # compute the mean log-likelihood
            Eq_eta *= mask_batch[:, :, None]
            ln_lik_[batch_idx] = (
                masked_logsumexp(Eq_eta, dim=1,
                                 mask=mask_batch[..., None].float())
                - mask_batch.sum(1)[:, None].log()
            )
            ln_prior_[batch_idx] = resp_batch
            Eq_ln_pi_[batch_idx] = doc_stick_batch['Eq_ln_pi']

            # free up some big variables to save mem
            del Eq_eta
            del temp_vars['zeta']
            del temp_vars['varphi']
            torch.cuda.empty_cache()

    return {
        'Eq_ln_eta': ln_lik_,
        'responsibility': ln_prior_,
        'Eq_ln_pi': Eq_ln_pi_,
        'w': params['w']
    }


def save_state(
    max_components_corpus: int,
    max_components_document: int,
    params: dict[str, torch.Tensor],
    share_alpha0: bool,
    out_path: str,
    prefix: str,
    it: int  # number of iteration so far
):
    """
    """
    ret = package_model(
        max_components_corpus,
        max_components_document,
        params, share_alpha0
    )
    path = Path(out_path) / f'{prefix}_it{it:d}.pkl'
    with path.open('wb') as fp:
        pkl.dump(ret, fp)


def variational_inference(
    dataset: Dataset,
    max_components_corpus: int,
    max_components_document: int,
    n_epochs: int,
    batch_size: int = 512,
    kappa: float = .5,
    tau0: float = 1.,
    batch_update: bool = False,
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
    ####################
    # Setup Dataloader
    ####################
    if warm_start_with and warm_start_with.whiten_params:
        dataset._whitening_params = warm_start_with.whiten_params

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
    it = params['start_iter']
    try:
        with tqdm(total=n_epochs, ncols=80, disable=not verbose) as prog:
            for _ in range(n_epochs):
                if batch_update:
                    reinit_ss(params)

                for mask_batch, data_batch, batch_idx in loader:

                    # send tensors to the target device
                    mask_batch = mask_batch.to(device)
                    data_batch = data_batch.to(device)
                    batch_idx = batch_idx.to(device)

                    with torch.no_grad():

                        # init some variables
                        # and re-init including accumulators for sufficient statistics
                        temp_vars = init_temp_vars(
                            max_components_corpus,
                            max_components_document,
                            data_batch,
                            mask_batch
                        )
                        if not batch_update:
                            reinit_ss(params)

                        # compute Eq_beta and probability
                        corpus_stick = compute_corpus_stick(
                            max_components_corpus,
                            params
                        )

                        # COMPUTE Eq[eta]
                        Eq_eta = compute_ln_p_phi_x(
                            data_batch,
                            mask_batch,
                            params['m'],
                            params['W'],
                            params['W_chol'],
                            params['W_logdet'],
                            params['W_isposdef'],
                            params['beta'],
                            params['nu']
                        )

                        # DO E-STEP AND COMPUTE BATCH LIKELIHOOD
                        noise_ratio = base_noise_ratio * 1000. / (it + 1000.)
                        ln_lik = e_step(
                            batch_idx, data_batch, mask_batch,
                            Eq_eta, params, temp_vars, corpus_stick,
                            share_alpha0 = share_alpha0,
                            e_step_tol = e_step_tol,
                            n_max_iter = n_max_inner_iter,
                            noise_ratio = noise_ratio,
                            eps = eps
                        )[0]

                        # COMPUTE LOWERBOUNDS
                        ln_lik *= len(dataset) / data_batch.shape[0]  # est. for the total lik
                        nw_prob = compute_normalwishart_probs(
                            params['m'], params['W'],
                            params['nu'], params['beta'],
                            params['W_chol'], params['W_logdet'],
                            params['m0'], params['W0_inv'],
                            params['nu0'], params['beta0']
                        )
                        total_ln_lik_est = (
                            ln_lik
                            + corpus_stick['Eq_ln_p_beta']
                            - corpus_stick['Eq_ln_q_beta']
                            + nw_prob
                        ).item()
                        params['training_lowerbound'].append(total_ln_lik_est)

                        if not batch_update:
                            # DO M-STEP
                            m_step(
                                batch_size, len(dataset),
                                params, corpus_stick, eps
                            )

                            # UPDATE NEW PARAMETERS
                            update_parameters(
                                cur_iter=it,
                                tau0=tau0,
                                kappa=kappa,
                                batch_size=batch_size,
                                params=params,
                                old_params=old_params,
                                batch_update=batch_update
                            )

                        # free up some big variables to save mem
                        del Eq_eta
                        del temp_vars['zeta']
                        del temp_vars['varphi']
                        torch.cuda.empty_cache()

                        if (
                            (save_every is not None) and
                            (save_every != 'epoch') and
                            (it % save_every == 0)
                        ):
                            save_state(
                               max_components_corpus,
                               max_components_document,
                               params, share_alpha0,
                               out_path, prefix, it
                            )

                    it += 1

                if batch_update:
                    with torch.no_grad():
                        # DO M-STEP
                        m_step(
                            len(dataset), len(dataset),
                            params, corpus_stick, eps
                        )

                        # UPDATE NEW PARAMETERS
                        update_parameters(
                            cur_iter=it,
                            tau0=tau0,
                            kappa=kappa,
                            batch_size=batch_size,
                            params=params,
                            old_params=old_params,
                            batch_update=batch_update
                        )

                if save_every == 'epoch':
                    save_state(
                       max_components_corpus,
                       max_components_document,
                       params, share_alpha0,
                       out_path, prefix, it
                    )

                prog.update()

    except KeyboardInterrupt as ke:
        print('User interrupted the training! finishing the fitting...')
    except Exception as e:
        raise e

    # wrap the result and return
    ret = package_model(
        max_components_corpus,
        max_components_document,
        params, share_alpha0
    )
    return ret
