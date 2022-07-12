from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import digamma
from scipy import sparse as sp


np_float = Union[np.float32, np.float64]
_LOG_2 = np.log(2.)
_LOG_2PI = np.log(2. * np.pi)


@dataclass
class Parameters:
    def log_prob(
        self,
        X: npt.NDArray[np_float]
    ) -> npt.NDArray[np_float]:
        """
        """
        raise NotImplementedError()


@dataclass
class NormalWishartParameters(Parameters):
    mu0: npt.NDArray[np_float]
    lmbda: float
    W: npt.NDArray[np_float]
    nu: float
    keep_inverse_W: bool=False
    W_inv: Optional[npt.NDArray[np_float]] = field(init=False)

    def __post_init__(self):
        """
        """
        d = self.mu0.shape[0]

        if self.keep_inverse_W:
            self.W_inv = np.linalg.inv(self.W)
        else:
            self.W_inv = None

        self.W_chol = np.linalg.cholesky(self.W)
        sign, self.logdet_W = np.linalg.slogdet(self.W)
        if sign <= 0.:
            raise ValueError(
                '[ERROR] log determinant of precision prior W is not defined!'
            )

        # first natural parameter
        # (the second one Eq[eta2] is computed directly using
        #  Cholesky decomposition of the precision prior)
        self.Eq_eta_1 = self.nu * self.W @ self.mu0  # (d,)

        # expected value of the log normalizer a(eta)
        range_d = np.arange(d, dtype=self.mu0.dtype)
        self.Eq_a_eta = (
            # .5 * self.nu * self.mu0.T @ self.W @ self.mu0
            .5 * self.Eq_eta_1 @ self.mu0  # equivalent to full computation above
            - .5 * (
                digamma(.5 * (self.nu - range_d)).sum()
                + d * _LOG_2
                + self.logdet_W  # we use "inverse wishart"
            )
        )

    def log_prob(
        self,
        X: npt.NDArray[np_float]
    ) -> npt.NDArray[np_float]:
        """
        """
        XW = X @ self.W_chol
        X_Eq_eta_2_X = -.5 * self.nu * np.square(XW).sum(1)
        return X @ self.Eq_eta_1 + X_Eq_eta_2_X - self.Eq_a_eta


@dataclass
class DirichletMultinomialParameters(Parameters):
    alphas: npt.NDArray[np_float]

    def __post_init__(self):
        """
        """
        # first natural parameter
        self.Eq_eta_1 = digamma(self.alphas) - digamma(self.alphas.sum())  # (d,)
        self.Eq_a_eta = 0.

    def log_prob(
        self,
        X: Union[npt.NDArray[np_float], sp.csr_matrix]
    ) -> npt.NDArray[np_float]:
        """
        """
        return X @ self.Eq_eta_1 - self.Eq_a_eta


@dataclass
class DPParameters(Parameters):
    max_components: int
    alphas: npt.NDArray[np_float]
    betas: npt.NDArray[np_float]

    def __post_init__(self):
        digamma_sum = digamma(self.alphas + self.betas)
        self.Eq_logV = (
            digamma(self.alphas) - digamma_sum
        )
        self.Eq_log_1_minus_V = digamma(self.betas) - digamma_sum
        self.Eq_log_1_minus_V_cumsum = (
            np.r_[0., np.cumsum(self.Eq_log_1_minus_V[:-1])]
        )
        self.log_pi_v = self.Eq_logV + self.Eq_log_1_minus_V_cumsum
        self.pi_v = np.exp(self.log_pi_v)


@dataclass
class BetaParameters(Parameters):
    alpha: float
    beta: float


@dataclass
class GammaParameters(Parameters):
    alpha: float  # shape
    beta: float  # inverse scale (rate)

    @property
    def mean(self):
        return self.alpha / self.beta
