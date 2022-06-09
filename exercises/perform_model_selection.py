from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    axis = np.linspace(-1.2, 2, n_samples)
    true_responses = f(axis)
    noises_responses = true_responses + np.random.normal(0, noise, n_samples)

    train_axis, train_responses, test_axis, test_responses = \
        split_train_test(pd.DataFrame(axis), pd.Series(noises_responses), 2 / 3)

    # Convert to np.array
    train_axis = np.array(train_axis[0]).reshape((train_axis.shape[0], 1))
    train_responses = np.array(train_responses)
    test_axis = np.array(test_axis[0]).reshape((test_axis.shape[0], 1))
    test_responses = np.array(test_responses)

    fig_q1 = make_subplots(rows=1, cols=1)
    fig_q1.add_trace(
        go.Scatter(x=axis, y=true_responses, mode='markers', marker_color="black", name="noiseless model")
    )
    fig_q1.add_trace(
        go.Scatter(x=train_axis.ravel(), y=train_responses, mode='markers', marker_color="blue", name="train set")
    )
    fig_q1.add_trace(
        go.Scatter(x=test_axis.ravel(), y=test_responses, mode='markers', marker_color="red", name="test set")
    )
    fig_q1.update_layout(height=600, width=1000,
                         title_text="Plots for noiseless model, train set ans test set",
                         xaxis_title="Samples",
                         yaxis_title="function Responses")
    fig_q1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    polynomial_degrees = np.linspace(0, 10, 11).astype(np.int_)
    training_errors = np.zeros(11)
    validation_errors = np.zeros(11)

    # Perform 5-Fold Cross-Validation
    for i in range(11):
        polynomial_model = PolynomialFitting(i)
        training_errors[i], validation_errors[i] = \
            cross_validate(polynomial_model, train_axis, train_responses, mean_square_error)

    fig_q2 = make_subplots(rows=1, cols=1)
    fig_q2.add_trace(
        go.Scatter(x=polynomial_degrees, y=training_errors, mode='lines', marker_color="blue", name="training error")
    )
    fig_q2.add_trace(
        go.Scatter(x=polynomial_degrees, y=validation_errors, mode='lines', marker_color="red", name="validation error")
    )
    fig_q2.update_layout(height=600, width=1000,
                         title_text="Training and Validation error as a function of the degree (Noise =  %.1f)" % noise,
                         xaxis_title="Polynomial degree",
                         yaxis_title="Error")
    fig_q2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_degree = np.argmin(validation_errors)
    polynomial_model = PolynomialFitting(best_degree)
    polynomial_model.fit(train_axis, train_responses)
    test_predicted_responses = polynomial_model.predict(test_axis)
    test_error = mean_square_error(test_responses, test_predicted_responses)
    print("Number of samples: %d, noise: %.2f" % (n_samples, noise))
    print("Polynomial degree with minimal validation error (%.2f): %d" % (validation_errors[best_degree], best_degree))
    print("Test error for polynomial model with degree %d: %.2f" % (best_degree, test_error))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
<<<<<<< Updated upstream
    # Question 1 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()
=======
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data, responses = datasets.load_diabetes(return_X_y=True)
    train_data = data[0:n_samples, :]
    train_responses = responses[0:n_samples]
    test_data = data[n_samples:, :]
    test_responses = responses[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_values = np.linspace(0.001, 3, n_evaluations)
    ridge_train_errors = np.zeros(n_evaluations)
    ridge_validation_errors = np.zeros(n_evaluations)
    lasso_train_errors = np.zeros(n_evaluations)
    lasso_validation_errors = np.zeros(n_evaluations)

    # Perform 5-Fold Cross-Validation
    for i in range(len(lambda_values)):
        ridge_model = RidgeRegression(lambda_values[i])
        lasso_model = Lasso(alpha=lambda_values[i], tol=1e-9, max_iter=10**5)
        ridge_train_errors[i], ridge_validation_errors[i] = \
            cross_validate(ridge_model, train_data, train_responses, mean_square_error)
        lasso_train_errors[i], lasso_validation_errors[i] = \
            cross_validate(lasso_model, train_data, train_responses, mean_square_error)

    # Plot for Ridge regression
    ridge_fig_q7 = make_subplots(rows=1, cols=1)
    ridge_fig_q7.add_trace(
        go.Scatter(x=lambda_values, y=ridge_train_errors, mode='lines', marker_color="blue", name="training error")
    )
    ridge_fig_q7.add_trace(
        go.Scatter(x=lambda_values, y=ridge_validation_errors, mode='lines', marker_color="red", name="validation error")
    )
    ridge_fig_q7.update_layout(height=600, width=1000,
                         title_text="Ridge Training and Validation errors as a function of the lambda values",
                         xaxis_title="Lambda value",
                         yaxis_title="Error")
    ridge_fig_q7.show()

    # Plot for Lasso regression
    lasso_fig_q7 = make_subplots(rows=1, cols=1)
    lasso_fig_q7.add_trace(
        go.Scatter(x=lambda_values, y=lasso_train_errors, mode='lines', marker_color="blue", name="training error")
    )
    lasso_fig_q7.add_trace(
        go.Scatter(x=lambda_values, y=lasso_validation_errors, mode='lines', marker_color="red", name="validation error")
    )
    lasso_fig_q7.update_layout(height=600, width=1000,
                               title_text="Lasso Training and Validation errors as a function of the lambda values",
                               xaxis_title="Lambda value",
                               yaxis_title="Error")
    lasso_fig_q7.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lambda_value = np.argmin(ridge_validation_errors)
    lasso_best_lambda_value = np.argmin(lasso_validation_errors)

    # Initialize each regression model
    linear_regression_model = LinearRegression()
    ridge_model = RidgeRegression(lambda_values[ridge_best_lambda_value])
    lasso_model = Lasso(alpha=lambda_values[lasso_best_lambda_value], tol=1e-9, max_iter=10**5)

    # Fit each regression model with the training data
    linear_regression_model.fit(train_data, train_responses)
    ridge_model.fit(train_data, train_responses)
    lasso_model.fit(train_data, train_responses)

    # Using each regression model, get predicted values for test data
    LR_predicted_test_responses = linear_regression_model.predict(test_data)
    ridge_predicted_test_responses = ridge_model.predict(test_data)
    lasso_predicted_test_responses = lasso_model.predict(test_data)

    # Get test errors (computed by MSE loss-function) for each model
    LR_test_error = mean_square_error(test_responses, LR_predicted_test_responses)
    ridge_test_error = mean_square_error(test_responses, ridge_predicted_test_responses)
    lasso_test_error = mean_square_error(test_responses, lasso_predicted_test_responses)

    # Print the test error for each model
    print("Test error for Least Square regression: %.2f" % LR_test_error)
    print("Test error for Ridge regression: %.2f" % ridge_test_error)
    print("Test error for Lasso regression: %.2f" % lasso_test_error)
>>>>>>> Stashed changes


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()

