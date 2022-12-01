from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import digamma
from scipy import sparse as sp


# some aliasings...
np_float = Union[np.float32, np.float64]
_LOG_2 = np.log(2.)
_LOG_2PI = np.log(2. * np.pi)


@dataclass
class Parameters:
    """
    It is a base dataclass that is inherited to diverse parameter objects.

    for some of the intermediate values (i.e., Eq_eta_1, Eq_eta_2, etc.),
    We followed (Blei and Jordan 2006) and Wikipedia entry of the `exponential family`_.

    .. _exponential family: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions

    """
    def log_prob(
        self,
        X: npt.NDArray[np_float]
    ) -> npt.NDArray[np_float]:
        """ compute log probability of given observation X

        Args:
            X: input data of shape of (n_samples, ...). Trailing sizes represents
               the shape of the objects

        Returns:
            an array of shape (n_samples,) contains log-probabilities of
            given set of observations.
        """
        raise NotImplementedError()


@dataclass
class NormalWishartParameters(Parameters):
    """
    It contains the parameters of Normal-Wishart distribution, namely
    location (m0), precision (W), the degree of freedom (nu), and the
    precision scaler (lmbda).

    Attributes:
        mu0 (:obj:`numpy.typing.NDArray`): location parameter
        lmbda (float): precision scale parameter
        W (:obj:`numpy.typing.NDArray`): precision parameter
        nu (:obj:`numpy.typing.NDArray`): degree of freedom parameter
        keep_inverse_W (bool): if set True, it computes the inverse of the
                               precision and keep it as an attributes (W_inv)
        W_chol (:obj:`numpy.typing.NDArray`):
            optional field contains the Cholesky decomposition of the precision
        logdet_W (float): log determinant of precision
        W_inv (:obj:`numpy.typing.NDArray`, optional):
            optional field contains the inverse of precision
        Eq_eta_1 (:obj:`numpy.typing.NDArray`):
            contains :math:`E_{q}[\\eta]`, which follows (Blei and Jordan 2006)
        Eq_a_eta (:obj:`numpy.typing.NDArray`):
            contains :math:`E_{q}[a(\\eta)]`, which follows (Blei and Jordan 2006)
        stable (bool): with this flag, the "square root" of the precision matrix W
                       is computed using the SVD with forcefully truncating the
                       eigenvalues larger or equal to zero. If not set, it is
                       computed by the Cholesky decomposition.
    """
    mu0: npt.NDArray[np_float]
    lmbda: float
    W: npt.NDArray[np_float]
    nu: float
    keep_inverse_W: bool=False
    W_inv: Optional[npt.NDArray[np_float]] = field(init=False)
    stable: bool=True

    def __post_init__(self):
        d = self.mu0.shape[0]

        if self.stable:
            U, s, _ = np.linalg.svd(self.W)
            # truncate the non-zero eigen values (dirty hack to make the
            # reconstructed W to be somehow positive semi-definitive
            s[s < 0] = 0.
            self.W_chol = U @ np.diag(np.sqrt(s))
            self.W = self.W_chol @ self.W_chol.T   # we do the cheating here as well

            # https://math.stackexchange.com/a/2001054
            self.logdet_W = np.log(np.maximum(
                np.diag(self.W_chol), np.finfo('float').eps * 10
            )).sum()

        else:
            self.W_chol = np.linalg.cholesky(self.W)

            sign, self.logdet_W = np.linalg.slogdet(self.W)
            if sign <= 0.:
                raise ValueError(
                    '[ERROR] log determinant of precision prior W is not defined!'
                )

        if self.keep_inverse_W:
            self.W_inv = np.linalg.inv(self.W)
        else:
            self.W_inv = None

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
        """ compute log probability of given observation X

        Args:
            X: input data of shape of (n_samples, ...). Trailing sizes represents
               the shape of the objects

        Returns:
            an array of shape (n_samples,) contains log-probabilities of
            given set of observations.
        """
        XW = X @ self.W_chol
        X_Eq_eta_2_X = -.5 * self.nu * np.square(XW).sum(1)
        return X @ self.Eq_eta_1 + X_Eq_eta_2_X - self.Eq_a_eta


@dataclass
class DirichletMultinomialParameters(Parameters):
    """
    It contains the parameters of Dirichlet-Multinomial distribution, namely
    alphas, which is the concentration parameter of Dicichlet prior.

    Attributes:
        alphas (:obj:`numpy.typing.NDArray`): concentration parameters
        Eq_eta_1 (:obj:`numpy.typing.NDArray`):
            contains :math:`E_{q}[\\eta]`, which follows (Blei and Jordan 2006)
        Eq_a_eta (:obj:`numpy.typing.NDArray`):
            contains :math:`E_{q}[a(\\eta)]`, which follows (Blei and Jordan 2006)
    """

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
        """ compute log probability of given observation X

        Args:
            X: input data of shape of (n_samples, ...). Trailing sizes represents
               the shape of the objects

        Returns:
            an array of shape (n_samples,) contains log-probabilities of
            given set of observations.
        """
        return X @ self.Eq_eta_1 - self.Eq_a_eta


@dataclass
class DPParameters(Parameters):
    """
    It contains the parameters of Dirichlet Process parameters, namely
    the shape parameter alphas and betas. The dimensionality of them is truncated to a
    finite integer `K`, as it mostly assumes the variational inference.

    Attributes:
        max_components (:obj:`int`): the maximum number of components (trunctation threshold)
        alphas (:obj:`numpy.typing.NDArray`): the first shape parameter of Beta distributions
        betas (:obj:`numpy.typing.NDArray`): the second shape parameter of Beta distributions
        Eq_logV (:obj:`numpy.typing.NDArray`):
            contains :math:`E_{q}[\\text{log}V]`, which follows (Blei and Jordan 2006)
        Eq_log_1_minus_V (:obj:`numpy.typing.NDArray`):
            contains :math:`E_{q}[\\text{log}(1 - V)]`, which follows (Blei and Jordan 2006)
        Eq_log_1_minus_V_cumsum (:obj:`numpy.typing.NDArray`):
            contains the cumulative sum of :math:`E_{q}[\\text{log}(1 - V)]` for handy
            computation.
        pi_v (:obj:`numpy.typing.NDArray`): it is :math:`E_{q}[\\text{log} \pi]` which is
            the sum of :math:`E_{q}[\\text{log}V]` and :math:`E_{q}[\\text{log}(1 - V)]`.
        log_pi_v (:obj:`numpy.typing.NDArray`): log of :math:`E_{q}[\\text{log} \pi]`
    """

    max_components: int
    alphas: npt.NDArray[np_float]
    betas: npt.NDArray[np_float]

    def __post_init__(self):
        """
        """
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
    """
    It contains the parameters of Beta distribution, `alpha` and `beta`.

    Attributes:
        alpha (float): the first shape parameter of Beta distribution
        beta (float): the second shape parameter of Beta distribution
    """

    alpha: float
    beta: float


@dataclass
class GammaParameters(Parameters):
    """
    It contains the parameters of Gamma distribution `alpha` and `beta`.

    Attributes:
        alpha (float): the shape parameter of Gamma distribution
        beta (float): the inverse scale parameter of Gamma distribution
    """

    alpha: float  # shape
    beta: float  # inverse scale (rate)

    @property
    def mean(self):
        """ compute the average of Gamma distribution

        Returns:
            computed average of the Gamma distribution
        """
        return self.alpha / self.beta
