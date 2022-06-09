from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    m = X.shape[0]
    set_size = int(m/cv)
    remnant = m - set_size * cv

    # Shaffle the indices
    random_indices = np.random.permutation(m)
    X = X[random_indices]
    y = y[random_indices]

    # Perform folding
    i = 0
    weight = 0
    train_score = 0
    validation_score = 0
    while i < m:
        # To keep each group size as even as possible
        if remnant > 0:
            remnant -= 1
            j = 1
        else:
            j = 0

        # Split for data and responses to training and test
        training_data = np.r_[X[:i, :], X[i + set_size + j:, :]]
        training_true_responses = np.r_[y[:i], y[i + set_size + j:]]
        test_data = X[i:i + set_size + j, :]
        test_true_responses = y[i:i + set_size + j]

        # Fit the estimator based on the folded training data
        estimator.fit(training_data, training_true_responses)
        # Predict the responses for the training data and test data
        training_predict_responses = estimator.predict(training_data)
        test_predict_responses = estimator.predict(test_data)

        # Compute the training and test predictions errors based on the scoring function
        train_error = scoring(training_true_responses, training_predict_responses)
        test_error = scoring(test_true_responses, test_predict_responses)

        # Update the averages for train_score and validation_score
        train_score = (train_score * weight + train_error) / (weight + 1)
        validation_score = (validation_score * weight + test_error) / (weight + 1)

        i += (set_size + j)
        weight += 1

    return train_score, validation_score
