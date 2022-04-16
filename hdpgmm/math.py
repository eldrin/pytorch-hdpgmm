from typing import Optional
import torch


def th_logdotexp(A, B):
    """
    numerically safer dot between
    exponentiated logged matrices and
    putting the result back to logarithm
    """
    max_A = torch.max(A)
    max_B = torch.max(B)
    C = torch.mm((A - max_A).exp(), (B - max_B).exp())
    torch.log(C, out=C)
    C += max_A + max_B
    return C


def th_batch_logdotexp_2d(A, B):
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
):
    """Numerically stable logsumexp on the last dim of `inputs`.
       reference: https://github.com/pytorch/pytorch/issues/2591
    Args:
        inputs: A Variable with any shape.
        keepdim: A boolean.
        mask: A mask variable of type float. It has the same shape as `inputs`.
              **ATTENTION** invalid entries are masked to **ONE**, not ZERO
    Returns:
        Equivalent of log(sum(exp(inputs), keepdim=keepdim)).

    NOTE: this function is based on this implementation:
        https://gist.github.com/pcyin/b027ffec9b1bc1b87ba02286b55c2484
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
