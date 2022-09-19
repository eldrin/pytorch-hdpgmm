from typing import Union, Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
# import librosa

import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F


class HDFMultiVarSeqDataset(Dataset):
    """
    It extends pytorch's `Dataclass` to handle documents with variable length
    sequence/bag-of-features. It assumes the preprocessed HDF file is stored
    and this class fetching the data dynamically from the file through
    the open connection. (per every call of `__getitem__`)

    Attributes:
        h5_fn (Union[str, :obj:`Path`]): path to the HDF file
        whiten (bool): if set True, the dataset precompute the relevant parameters
                       at initialization and whitening the data.
        chunk_size (:obj:`int`): it is used to determine the length of the data when
                          whitening parameters are computed to save memory.
        verbose (bool): if set True, the progress of computing parameters for
                        whitening is reported in the standard output.
    """
    def __init__(
        self,
        h5_fn: Union[str, Path],
        whiten: bool=False,
        chunk_size: int=1024,
        verbose: bool=False
    ):
        self.h5_fn = h5_fn
        self.whiten = whiten
        self.chunk_size = chunk_size
        self.verbose = verbose

        if whiten:
            self._init_whitening()

        # cache the size of the dataset
        with h5py.File(h5_fn, 'r') as hf:
            self._length = hf['indptr'].shape[0] - 1
            self._raw_nrow, self.dim = hf['data'].shape
            self.ids = hf['ids'][:]

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self,
        idx: int
    ) -> tuple[int, torch.Tensor]:
        """ fetch a single document

        calling this, one can get document. (variable length, sequence of multivariate vectors)

        Args:
            idx: index for the document to be fetched

        Returns:
            a tuple of index and the tensor corresponding to given input index
        """
        with h5py.File(self.h5_fn, 'r') as hf:
            # index frames/tokens
            j0, j1 = hf['indptr'][idx], hf['indptr'][idx+1]
            x = hf['data'][j0:j1]

        # whiten, if needed
        x = self.apply_whitening(x)

        # wrap to torch.Tensor
        x = torch.as_tensor(x, dtype=torch.float32)

        return (idx, x)

    def apply_whitening(
        self,
        x: npt.ArrayLike
    ) -> npt.ArrayLike:
        """ apply whitening to the input data x

        Args:
            x: data fetched to be processed, from HDF file

        Returns:
            whitened tensor
        """
        if self.whiten:
            x -= self._whitening_params['mean'][None]
            x = x @ self._whitening_params['precision_cholesky']
        return x

    def _init_whitening(self):
        """ initialize whitening parameters

        it computes parameters needed for whitening process using given dataset.
        """
        # compute whitening parameters
        with h5py.File(self.h5_fn, 'r') as hf:
            self._whitening_params = compute_global_mean_cov(hf,
                                                             self.chunk_size,
                                                             self.verbose)


class AudioDataset(Dataset):
    """
    """
    def __init__(
        self,
        audio_list_fn: Union[str, Path],
        transform: Optional[Callable] = None,
        chunk_size: int=1024,
        verbose: bool=False
    ):
        """
        """
        # load audio list
        if isinstance(audio_list_fn, str):
            audio_list_fn = Path(audio_list_fn)

        with audio_list_fn.open() as fp:
            self._audio_fns = [l.replace('\n', '') for l in fp]
        self.audio_list_fn = audio_list_fn

        self.transform = transform

    def __len__(self) -> int:
        return len(self._audio_fns)

    def __getitem__(
        self,
        idx: int
    ) -> tuple[int, torch.Tensor]:
        """
        """
        y, sr = torchaudio.load(self._audio_fns[idx])
        if y.shape[0] > 1:
            y = y.mean(0)
        if sr != 22050:
            y = F.resample(y, sr, 22050)

        if self.transform:
            x = self.transform(y.numpy(), sr)
            if len(x.shape) > 2:
                # sometimes it comes with weird trailing dimension
                # so we index it forcefully
                x = x[..., 0]  # this maybe a bug...

        # wrap to torch.Tensor
        x = torch.as_tensor(x, dtype=torch.float32)

        return (idx, x)


def collate_var_len_seq(
    samples: list[tuple[int, torch.Tensor]],
    max_len: Optional[int] = None,
) -> tuple[torch.BoolTensor,      # mask_batch
           torch.Tensor,      # data_batch
           torch.LongTensor]: # batch_idx
    """ collate variable length sequence / bag-of-features into masked tensor

    Args:
        samples: list of indices and corresponding variable length documents to be collated
        max_len: threshold to cut off the substantially long sequence

    Returns:
        tuple of processed tensors (i.e., mask, data, indices).
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
    """ compute global gaussian statistics (mean and covariance)

    It also computes precision and its Cholesky decomposition, which are
    useful for some of the other computations involved in the HDPGMM VI inference.

    Args:
        hf: HDF file object via h5py.File
        chunk_size: it is used to determine the length of the data when
                    whitening parameters are computed to save memory.
        verbose: if set True, the progress of computing parameters for
                 whitening is reported in the standard output.

    Returns:
        computed statistics stored in a dictionary. the keys are the
        name of the statistics (i.e., mean) and values are tensor containing
        the computed values.
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
