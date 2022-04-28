from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        # Create class_to_samples dict and nk_sum vector
        k = self.classes_.shape[0]
        class_to_samples = {}
        nk_sum = np.zeros(k, dtype=int)
        for t in range(k):
            class_to_samples[t] = []
            for i in range(m):
                if y[i] == self.classes_[t]:
                    class_to_samples[t].append(i)
                    nk_sum[t] += 1

        # Fit self.pi_
        self.pi_ = (1 / m) * nk_sum

        # Fit self.mu_
        for t in range(k):
            mu_t = np.zeros(d, dtype=float)
            for i in class_to_samples[t]:
                mu_t += X[i]

            mu_t = (1 / nk_sum[t]) * mu_t
            if t == 0:
                self.mu_ = np.array([mu_t])
            else:
                self.mu_ = np.r_[self.mu_, [mu_t]]

        # Fit self.vars_
        self.vars_ = np.zeros((k, d), dtype=float)
        for t in range(k):
            variances = np.zeros(d, dtype=float)
            for i in class_to_samples[t]:
                variances += np.power(X[i] - self.mu_[t], 2)
            variances = (1 / nk_sum[t]) * variances

            if t == 0:
                self.vars_ = np.array([variances])
            else:
                self.vars_ = np.r_[self.vars_, [variances]]

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
        k = self.classes_.shape[0]
        # Construct sum_k vector (n_classes) for better performance
        sum_k = np.zeros(k, dtype=float)
        for t in range(k):
            sum_k[t] = np.log(self.pi_[t]) - 0.5 * np.sum(np.log(self.vars_[t]))

        m = X.shape[0]
        responses = np.zeros(m, dtype=float)
        for i in range(m):
            vector_i = np.zeros(k, dtype=float)
            for t in range(k):
                sum2 = - 0.5 * np.sum(np.power(X[i] - self.mu_[t], 2) / self.vars_[t])
                vector_i[t] = sum_k[t] + sum2

            responses[i] = self.classes_[np.argmax(vector_i)]

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
        for i in range(m):
            for t in range(k):
                coefficient = 1 / np.power(np.pow(2 * np.pi, d) * np.prod(self.vars_[t]), 0.5)
                exp = np.exp(- 0.5 * np.sum(np.power(X[i] - self.mu_[t], 2) / self.vars_[t]))
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
