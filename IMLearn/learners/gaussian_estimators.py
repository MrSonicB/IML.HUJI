from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # Estimated expectation
        self.mu_ = np.mean(X)
        # Estimated variance
        if self.biased_:
            self.var_ = np.mean(np.power(X - self.mu_, 2))
        else:
            self.var_ = np.sum(np.power(X - self.mu_, 2)) / (len(X) - 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        coefficient = 1 / (np.power(2 * np.pi * self.var_, 0.5))
        exp = np.exp((-1 * np.power(X - self.mu_, 2) / (2 * self.var_)))
        return coefficient * exp


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # After deriving the log-likelihood function, I got the following:
        num_of_samples = X.shape[0]
        sum1 = num_of_samples * np.log(2 * np.pi * sigma)
        sum2 = (1 / np.power(sigma, 2)) * np.sum(np.power(X - mu, 2))
        return -0.5 * (sum1 + sum2)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # Estimated expectation vector
        self.mu_ = np.mean(X, axis=0)
        # Estimated covariance matrix
        tmp_matrix = X
        for j in range(X.shape[1]):
            tmp_matrix[:, j] = tmp_matrix[:, j] - self.mu_[j]
        self.cov_ = (1 / (X.shape[0] - 1)) * (np.transpose(tmp_matrix) @ tmp_matrix)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        def compute_pdf_for_vector(X1: np.ndarray):
            """
            Parameters
            ----------
            X1: ndarray of shape (n_features, ) Sample to calculate PDF for

            Returns
            ----------
            Calculated PDF value (float) for the given sample
            """
            d = self.mu_.shape[0]
            coefficient = 1 / (np.power(np.power(2*np.pi, d) * np.abs(det(self.cov_)), 0.5))
            exp = np.exp(-0.5 * (np.transpose(X1 - self.mu_) @ (np.linalg.inv(self.cov_) @ (X1 - self.mu_))))
            return coefficient * exp

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        return np.apply_along_axis(compute_pdf_for_vector, 1, X)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        # Just to save running time
        cov_inverse = inv(cov)

        def compute_multiplication(X1: np.ndarray):
            """
            Parameters
            ----------
            X1: ndarray of shape (n_features, ) Sample to calculate PDF for

            Returns
            ----------
            Calculated value (float) of: (X1 - mu)^t * cov^-1 * (X1 - mu)
            """
            return np.transpose(X1 - mu) @ (cov_inverse @ (X1 - mu))

        # After deriving the log-likelihood function, I got the following:
        num_of_samples = X.shape[0]
        num_of_features = X.shape[1]
        sum1 = num_of_samples * (num_of_features * np.log(2 * np.pi) + np.log(np.abs(det(cov))))
        sum2 = np.sum(np.apply_along_axis(compute_multiplication, 1, X))
        return -0.5 * (sum1 + sum2)
