from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        d = X.shape[1]

        # Fit self.classes_
        self.classes_ = np.unique(y)

        # Create sample_to_class dict and nk_sum vector
        k = self.classes_.shape[0]
        sample_to_class = {}
        nk_sum = np.zeros(k, dtype=int)
        for i in range(m):
            for t in range(k):
                if y[i] == self.classes_[t]:
                    sample_to_class[i] = t
                    nk_sum[t] += 1
                    break

        # Fit self.pi_
        self.pi_ = (1 / m) * nk_sum

        # Fit self.mu_
        for t in range(k):
            mu_t = np.zeros(d, dtype=float)
            for i in range(m):
                if sample_to_class[i] == t:
                    mu_t += X[i]

            mu_t = (1 / nk_sum[t]) * mu_t
            if t == 0:
                self.mu_ = np.array([mu_t])
            else:
                self.mu_ = np.r_[self.mu_, [mu_t]]

        # Fit self.cov_
        self.cov_ = np.zeros((d, d), dtype=float)
        for i in range(m):
            vector = X[i] - self.mu_[sample_to_class[i]]
            self.cov_ += np.multiply(vector.reshape(d, 1), vector)

        self.cov_ = (1 / m) * self.cov_

        # Fot self._cov_inv
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # Construct A matrix (n_classes,n_features), and b vector (n_classes,) for better performance
        k = self.classes_.shape[0]
        a_0 = np.dot(self._cov_inv, self.mu_[0])
        b_0 = np.log(self.pi_[0]) - 0.5 * np.dot(self.mu_[0], a_0)
        A = np.array([a_0])
        b = np.array([b_0])

        for t in range(1, k):
            a_t = np.dot(self._cov_inv, self.mu_[t])
            b_t = np.log(self.pi_[t]) - 0.5 * np.dot(self.mu_[t], a_t)
            A = np.r_[A, [a_t]]
            b = np.r_[b, [b_t]]

        # Calculate response for each sample
        m = X.shape[0]
        responses = np.zeros(m, dtype=float)
        for i in range(m):
            vector_k = np.dot(A, X[i]) + b
            responses[i] = self.classes_[np.argmax(vector_k)]

        return responses

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m = X.shape[0]
        d = X.shape[1]
        k = self.classes_.shape[0]
        likelihood_matrix = np.zeros((m, k), dtype=float)
        coefficient = 1 / np.power(np.pow(2 * np.pi, d) * np.linalg.det(self.cov_), 0.5)
        for i in range(m):
            for t in range(k):
                vector_i_t = X[i] - self.mu_[t]
                exp = np.exp(-0.5 * np.dot(vector_i_t, np.dot(self._cov_inv, vector_i_t)))
                likelihood_matrix[i][t] = coefficient * exp

        return likelihood_matrix


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
