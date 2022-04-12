from typing import Union, Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt

import h5py

import torch
from torch.utils.data import Dataset


class HDFMultiVarSeqDataset(Dataset):
    def __init__(self,
                 h5_fn: Union[str, Path]):
        """
        """
        self.h5_fn = h5_fn
        self._hf = h5py.File(h5_fn, 'r')

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
        j0, j1 = self._hf['indptr'][idx], self._hf['indptr'][idx+1]
        x = torch.as_tensor(self._hf['data'][j0:j1],
                            dtype=torch.float32)
        return (idx, x)


def collate_var_len_seq(
    samples: list[tuple[int, torch.Tensor]],
    max_len: Optional[int] = None
) -> tuple[torch.Tensor,      # mask_batch
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

    mask = torch.zeros((batch_size, max_len_batch), dtype=torch.float32)
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


def draw_mini_batch(
    batch_size: int,
    indptr: npt.ArrayLike,
    data: npt.ArrayLike,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    """
    # draw mini-batch
    batch_idx = np.random.choice(len(indptr) - 1, batch_size, False)

    # slice the batch
    indptr_batch = [0]
    data_batch = []
    for j in batch_idx:
        j0, j1 = indptr[j], indptr[j+1]
        indptr_batch.append(indptr_batch[-1] + j1 - j0)
        data_batch.append(data[j0:j1])
    data_batch = np.concatenate(data_batch)

    indptr_batch_tch = torch.as_tensor(indptr_batch,
                                       dtype=torch.int64,
                                       device=device)
    data_batch_tch = torch.as_tensor(data_batch,
                                     dtype=torch.float32,
                                     device=device)

    max_len = max(indptr_batch_tch[1:] - indptr_batch_tch[:-1])
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool,
                       device=device)
    data_batch_mat = torch.zeros((batch_size, max_len, data.shape[-1]),
                                 dtype=torch.float32, device=device)
    for j in range(batch_size):
        j0, j1 = indptr_batch_tch[j], indptr_batch_tch[j+1]
        mask[j, :j1-j0] = 1.
        data_batch_mat[j, :j1-j0] = data_batch_tch[j0:j1]

    return mask, data_batch_mat, batch_idx
