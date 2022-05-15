from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        d = X.shape[1]
        signs = [-1, 1]
        minimum_error = 1

        for j in range(d):
            for sign in signs:
                tmp_threshold, tmp_error = self._find_threshold(X[:, j], y, sign)
                if tmp_error < minimum_error:
                    minimum_error = tmp_error
                    self.threshold_ = tmp_threshold
                    self.j_ = j
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        m = X.shape[0]
        res = []
        for i in range(m):
            if X[i, self.j_] >= self.threshold_:
                res.append(self.sign_)
            else:
                res.append(-1 * self.sign_)

        return np.array(res)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        m = values.shape[0]
        labels2values = np.zeros((2, m), dtype=float)
        labels2values[0, :] = labels
        labels2values[1, :] = values
        # Sort lab2les according to values
        labels2values = labels2values[:, labels2values[1, :].argsort()]

        threshold = labels2values[1, 0]
        tmp_labels = sign * np.ones(m, dtype=float)
        minimum_error = np.inner(np.abs(labels2values[0, :]), 1 - (tmp_labels == np.sign(labels2values[0, :])))
        tmp_error = minimum_error

        for i in range(m):
            if np.sign(labels2values[0, i]) == -1 * sign:
                tmp_error -= np.abs(labels2values[0, i])
            else:
                tmp_error += np.abs(labels2values[0, i])

            if tmp_error < minimum_error:
                minimum_error = tmp_error
                if i < m - 1:
                    threshold = labels2values[1, i + 1]
                else:
                    threshold = labels2values[1, m - 1] + 1

        return threshold, minimum_error

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
        return misclassification_error(y, self._predict(X))
