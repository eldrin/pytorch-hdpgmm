from typing import Optional
import torch


def th_logdotexp(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    numerically safer dot between exponentiated logged matrices and
    putting the result back to logarithm

    Args:
        A: left tensor
        B: right tensor

    Returns:
        resulted tensor of logdotexp
    """
    max_A = torch.max(A)
    max_B = torch.max(B)
    C = torch.mm((A - max_A).exp(), (B - max_B).exp())
    torch.log(C, out=C)
    C += max_A + max_B
    return C


def th_batch_logdotexp_2d(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    numerically safer dot between exponentiated logged matrices and
    putting the result back to logarithm. This function support the
    `batchdot` equivalent routine for logdotexp. The first axis represents
    the number of matrices of the targets of logdotexp

    Args:
        A: left tensor
        B: right tensor

    Returns:
        resulted tensor of batch logdotexp
    """
    max_A = torch.amax(A, dim=(1, 2))[:, None, None]
    max_B = torch.amax(B, dim=(1, 2))[:, None, None]
    C = torch.bmm((A - max_A).exp(), (B - max_B).exp())
    torch.log(C, out=C)
    C += max_A + max_B
    return C


def th_masked_logsumexp(
    inputs: torch.Tensor,
    dim: int=-1,
    keepdim: bool=False,
    mask: Optional[torch.Tensor]=None,
    max_offset_thrd: float=-1e10
) -> torch.Tensor:
    """ Numerically stable logsumexp on the last dim of `inputs`.

    reference: https://github.com/pytorch/pytorch/issues/2591

    NOTE: this function is based on this implementation:
        https://gist.github.com/pcyin/b027ffec9b1bc1b87ba02286b55c2484

    Args:
        inputs: A Variable with any shape.
        keepdim: A boolean.
        mask: A mask variable of type float. It has the same shape as `inputs`.
              **ATTENTION** invalid entries are masked to **ONE**, not ZERO
    Returns:
        Equivalent of log(sum(exp(inputs), keepdim=keepdim)).
    """

    if mask is not None:
        mask = 1. - mask
        max_offset = max_offset_thrd * mask
    else:
        max_offset = 0.

    s, _ = torch.max(inputs + max_offset, dim=dim, keepdim=True)

    inputs_offset = inputs - s
    if mask is not None:
        inputs_offset.masked_fill_(mask.bool(), -float('inf'))

    outputs = s + inputs_offset.exp().sum(dim=dim, keepdim=True).log()

    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def mat_sqrt(
    A: torch.Tensor,
    method: str = 'cholesky',
    ensure_pos_semidef: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor,
           torch.Tensor,
           torch.bool,
           torch.float]:
    """ Compute "square root" of the given symmetric matrix

    It computes the "square root" of the given symmetric matrix. It provides
    several methods, including the Cholesky decomposition (default), SVD,
    and finally eigen decomposition. The implementation is based on `here`_.

    Args:
        A: input real symmetric matrix
        method: specifying which method is used. {'cholesky', 'svd', 'eig'}
        ensure_pos_semidef: if True, it tries to fix the issue by some tricks.
                            however most of the case it should be used very carefully
                            otherwise it'll just produce a degenerated outcome

    Returns:
        A: the input matrix A, which could be reconstructed from adjusted square root matrix of it
        L: the decomposition of the input matrix which by computing the dot product
           of itself can recover the input matrix
        isposdef: boolean indicate the given matrix is positive semi definite
        abslogdet: absolute log determinant computed from :obj:`torch.linalg.slogdet`

    Raises:
        ValueError if method is not supported

    .. _here: https://nl.mathworks.com/matlabcentral/answers/225177-chol-gives-error-for-a-barely-positive-definite-matrix#answer_332354
    """
    if method not in {'cholesky', 'svd', 'eig'}:
        raise ValueError(
            '[ERROR] only `cholesky`, `svd` and `eig` methods are supported!'
        )

    if method == 'cholesky':
        L, info = torch.linalg.cholesky_ex(A)
        is_bad = info.sum() != 0
        if is_bad and ensure_pos_semidef:
            L, info = torch.linalg.cholesky_ex(A + eps * torch.eye(A.shape[-1]))

    elif method == 'eig':
        U, d = torch.linalg.eig(A)
        bads = d.float() < 0.
        is_bad = bads.sum() > 0
        if is_bad and ensure_pos_semidef:
            d[d.float() < 0] = 0.
        L = torch.linalg.qr(torch.diag(torch.sqrt(d)) @ U.mT).R.mT

    elif method == 'svd':
        U, s, _ = torch.linalg.svd(A)
        bads = s < 0.
        is_bad = bads.sum() > 0.
        if is_bad and ensure_pos_semidef:
            s[s < 0] = 0.
        L = torch.bmm(U, torch.diag_embed(torch.sqrt(s)))

    if ensure_pos_semidef:
        A = torch.bmm(L, L.mT)
    sign, logabsdet = torch.linalg.slogdet(A)
    if any(sign <= 0):
        raise ValueError('[ERROR] log det of W must be positive!')

    return A, L, is_bad, logabsdet
