from typing import Union, Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt

import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class HDFMultiVarSeqDataset(Dataset):
    def __init__(self,
                 h5_fn: Union[str, Path],
                 whiten: bool=False,
                 chunk_size: int=1024,
                 verbose: bool=False):
        """
        """
        self.h5_fn = h5_fn
        self._hf = h5py.File(h5_fn, 'r')
        self.whiten = whiten
        self.chunk_size = chunk_size
        self.verbose = verbose
        if whiten:
            self._init_whitening()

    def __del__(self):
        self._hf.close()

    def __len__(self) -> int:
        return self._hf['indptr'].shape[0] - 1

    def __getitem__(
        self,
        idx: int
    ) -> tuple[int, torch.Tensor]:
        """
        """
        # index frames/tokens
        j0, j1 = self._hf['indptr'][idx], self._hf['indptr'][idx+1]
        x = self._hf['data'][j0:j1]

        # whiten, if needed
        if self.whiten:
            x -= self._whitening_params['mean'][None]
            x = x @ self._whitening_params['precision_cholesky']

        # wrap to torch.Tensor
        x = torch.as_tensor(x, dtype=torch.float32)

        return (idx, x)

    def _init_whitening(self):
        """
        """
        # compute whitening parameters
        self._whitening_params = compute_global_mean_cov(self._hf,
                                                         self.chunk_size,
                                                         self.verbose)


def collate_var_len_seq(
    samples: list[tuple[int, torch.Tensor]],
    max_len: Optional[int] = None
) -> tuple[torch.BoolTensor,      # mask_batch
           torch.Tensor,      # data_batch
           torch.LongTensor]: # batch_idx
    """
    """
    if max_len is None:
        max_len = torch.inf

    batch_idx, samples = zip(*samples)
    batch_idx = torch.LongTensor(batch_idx)
    batch_size = len(samples)
    dim = samples[0].shape[-1]
    max_len_batch = min(max([s.shape[0] for s in samples]), max_len)

    mask = torch.zeros((batch_size, max_len_batch)).bool()
    data_batch_mat = torch.zeros((batch_size, max_len_batch, dim),
                                 dtype=torch.float32)

    for j, x in enumerate(samples):
        n = x.shape[0]
        if n > max_len:
            # randomly slice the data
            start = np.random.randint(n - max_len)
            mask[j] = 1.
            data_batch_mat[j] = x[start:start + max_len]
        else:
            mask[j, :n] = 1.
            data_batch_mat[j, :n] = x

    return mask, data_batch_mat, batch_idx


def compute_global_mean_cov(
    hf: h5py.File,
    chunk_size: int=1024,
    verbose: bool=False
) -> dict[str, npt.ArrayLike]:
    """
    """
    n = hf['data'].shape[0]
    x_sum = 0.
    xx_sum = 0.
    n_chunks = n // chunk_size + (n % chunk_size != 0)
    with tqdm(total=n_chunks, ncols=80, disable=not verbose) as prog:
        for i in range(n_chunks):
            x = hf['data'][i*chunk_size:(i+1)*chunk_size]
            x_sum += x.sum(0)
            xx_sum += x.T @ x
            prog.update()
        mean = x_sum / n
        cov = xx_sum / n - np.outer(mean, mean)
        prec = np.linalg.inv(cov)
        prec_chol = np.linalg.cholesky(prec)

    return {
        'mean': mean,
        'cov': cov,
        'precision': prec,
        'precision_cholesky': prec_chol
    }
