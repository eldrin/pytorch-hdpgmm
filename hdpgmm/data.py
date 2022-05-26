from typing import Union, Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import librosa

import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F


class HDFMultiVarSeqDataset(Dataset):
    def __init__(
        self,
        h5_fn: Union[str, Path],
        whiten: bool=False,
        chunk_size: int=1024,
        verbose: bool=False
    ):
        """
        """
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

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self,
        idx: int
    ) -> tuple[int, torch.Tensor]:
        """
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

    def apply_whitening(self, x):
        """
        """
        if self.whiten:
            x -= self._whitening_params['mean'][None]
            x = x @ self._whitening_params['precision_cholesky']
        return x

    def _init_whitening(self):
        """
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
    min_len: Optional[int] = 128
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


def ext_mel(
    y: npt.ArrayLike,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    db_scale: bool = True
) -> npt.ArrayLike:
    """
    """
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
    )
    if db_scale:
        m = librosa.power_to_db(m)
    return m.T


def ext_feature(
    y: npt.ArrayLike,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
    features: set[str] = {'mfcc', 'dmfcc', 'ddmfcc', 'chroma'}
) -> npt.ArrayLike:
    """
    """
    assert any([
        feature not in {'mfcc', 'dmfcc', 'ddmfcc', 'chroma'}
        for feature in features
    ])
    assert len(features) > 0

    m = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
    )

    features_ = []
    if 'mfcc' in features:
        features_.append(m)

    if 'dmfcc' in features:
        dm = librosa.feature.delta(m, order=1)
        features_.append(dm)

    if 'ddmfcc' in features:
        ddm = librosa.feature.delta(m, order=2)
        features_.append(ddm)

    if 'chroma' in features:
        chrm = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features_.append(chrm)

    # concatenate features
    if len(features_) > 1:
        features_ = np.concatenate(features_, axis=1)
    else:
        features_ = features_[0]

    return features_.T  # (n_frames/n_tokens, dim)


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
