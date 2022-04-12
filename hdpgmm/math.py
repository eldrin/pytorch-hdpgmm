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
