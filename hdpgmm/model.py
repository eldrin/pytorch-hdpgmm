from typing import Optional
from functools import partial
from pathlib import Path
import pickle as pkl

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from bibim.hdp import gaussian as hdpgmm
from bibim.data import MVVarSeqData

from .data import (HDFMultiVarSeqDataset,
                   collate_var_len_seq)


CORPUS_LEVEL_PARAMS =  {'m', 'C', 'beta', 'nu', 'u', 'v', 'y', 'w2'}
_LOG_2 = torch.log(torch.as_tensor(2.))
_LOG_PI = torch.log(torch.as_tensor(torch.pi))
_LOG_2PI = _LOG_2 + _LOG_PI


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
    batch_idx: torch.Tensor,
    data_batch: torch.Tensor,
    mask_batch: torch.Tensor,
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
        # zeta[:] = (
        #     Eq_ln_pi[:, None]
        #     + torch.bmm(Eq_eta_batch, torch.exp(varphi).permute(0, 2, 1))
        # )
        zeta -= torch.logsumexp(zeta, dim=-1)[:, :, None]
        zeta.copy_(
            torch.clamp(torch.exp(zeta), min=eps)
        )
        # zeta[:] = torch.clamp(torch.exp(zeta), min=eps)
        zeta *= mask_batch[:, :, None]

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
            + (zeta * torch.log(torch.clamp(zeta, min=eps)) * mask_batch[:, :, None]).sum()
        )
        ln_p_z += (zeta
                   * torch.bmm(Eq_eta_batch, torch.exp(varphi).permute(0, 2, 1))
                   * mask_batch[:, :, None]).sum()
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

    r = zeta @ torch.clamp(torch.exp(varphi), min=eps)
    if noise_ratio > 0:
        norm = r.sum(-1)
        r /= torch.clamp(norm[:, :, None], min=eps)
        e = torch.rand(*r.shape, device=data_batch.device)
        e /= e.sum(-1)[:, :, None]
        e *= mask_batch[:, :, None]
        r *= (1. - noise_ratio)
        r += e * noise_ratio
        r *= norm[:, :, None] * mask_batch[:, :, None]
    N += r.sum((0, 1))

    for k in range(K):
        x_bar[k] += torch.bmm(r[:, :, k][:, None], data_batch).sum(0)[0]
        rx = r[:, :, k][:, :, None]**.5 * data_batch
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
        old_params[name].copy_(params[name][:])

    # re-compute auxiliary variables
    W = torch.linalg.inv(params['C'] * params['nu'][:, None, None])
    params['W_chol'] = torch.linalg.cholesky(W)
    params['W_logdet'] = torch.linalg.slogdet(W).logabsdet

    # to compute the full Eq[log|lambda_k|]
    K, D = params['m'].shape
    arangeD = torch.arange(D, device=params['m'].device)
    log2 = torch.log(torch.as_tensor([2], device=params['m'].device))
    for k in range(K):
        params['W_logdet'][k] += (
            torch.digamma((params['nu'][k] - arangeD) * .5).sum()
            + D * log2.item()
        )


def init_params(
    max_components_corpus: int,
    max_components_documents: int,
    dataset: HDFMultiVarSeqDataset,
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
    warm_start_with: Optional[hdpgmm.HDPGMM] = None
) -> tuple[dict[str, torch.Tensor],
           dict[str, torch.Tensor]]:
    """
    """
    # get some vars set
    K = max_components_corpus
    T = max_components_documents

    # build temporary MVVarSeqData from hdf file pointer
    mvvarseqdat = MVVarSeqData(dataset._hf['indptr'][:],
                               dataset._hf['data'],
                               dataset._hf['ids'][:])

    # set / load / sample the hyper priors
    # TODO: this can be slow if the dataset get larger
    #       torch-gpu implementation could sped up this routine
    m0, W0, beta0, nu0 = hdpgmm.init_hyperprior(mvvarseqdat,
                                                m0, W0, nu0, beta0,
                                                n_W0_cluster, cluster_frac,
                                                warm_start_with=warm_start_with)

    # get initialization
    params = hdpgmm.initialize_params(K, T, mvvarseqdat,
                                      m0=m0, W0=W0, nu0=nu0, beta0=beta0,
                                      s1=s1, s2=s2, g1=g1, g2=g2,
                                      full_uniform_init=full_uniform_init,
                                      warm_start_with=warm_start_with)
    (m, C, nu, beta, W_chol, W_logdet,
     u, v, y, a, b, w, w2,
     start_iter, mean_holdout_probs, train_lik) = params

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
) -> hdpgmm.HDPGMM:
    """
    """
    return hdpgmm.package_model(
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
    )


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
    data_parallel_num_workers: int = 0,
    n_W0_cluster: int = 128,
    cluster_frac: float = .01,
    warm_start_with: Optional[hdpgmm.HDPGMM] = None,
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
    dataset = HDFMultiVarSeqDataset(h5_fn)
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
        max_components_corpus, max_components_document, dataset,
        m0, W0, nu0, beta0, s1, s2, g1, g2,
        device, n_W0_cluster, cluster_frac,
        full_uniform_init=full_uniform_init,
        warm_start_with=warm_start_with
    )

    #######################
    # Update loop!
    #######################
    trn_liks = []
    mean_holdout_probs = []
    it = 0
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
