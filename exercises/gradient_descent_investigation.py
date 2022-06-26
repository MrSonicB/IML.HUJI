import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                     go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    objective_values_list = []
    weights_list = []

    def callback_function(solver=None, weights=None, val=None, grad=None, t=None, eta=None, delta=None):
        objective_values_list.append(val)
        weights_list.append(weights)

    return callback_function, objective_values_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_fig = make_subplots(rows=1, cols=1)
    l2_fig = make_subplots(rows=1, cols=1)
    colors = ["blue", "orange", "green", "red"]
    for i in range(len(etas)):
        # L1 objective
        l1_module = L1(weights=init)
        l1_callback_func, l1_objective_values_list, l1_weights_list = get_gd_state_recorder_callback()
        l1_GD = GradientDescent(learning_rate=FixedLR(base_lr=etas[i]), out_type='best', callback=l1_callback_func)
        l1_solution = l1_GD.fit(f=l1_module, X=np.empty((0, 0)), y=np.empty(0))

        # L2 objective
        l2_module = L2(weights=init)
        l2_callback_func, l2_objective_values_list, l2_weights_list = get_gd_state_recorder_callback()
        l2_GD = GradientDescent(learning_rate=FixedLR(base_lr=etas[i]), out_type='best', callback=l2_callback_func)
        l2_solution = l2_GD.fit(f=l2_module, X=np.empty((0, 0)), y=np.empty(0))

        # Question 1 - Plotting the descent trajectory for L1 and L2 in case rate = 0.001
        if etas[i] == 0.01:
            # Plot L1 descent path
            plot_descent_path(module=L1, descent_path=np.array(l1_weights_list),
                              title="L1 module's descent path of GD with fixed LR").show()

            # Plot L2 descent path
            plot_descent_path(module=L2, descent_path=np.array(l2_weights_list),
                              title="L2 module's descent path of GD with fixed LR").show()

        # Question 3 - Plotting the convergence rates for L1 and L2 in each rate

        # L1 norm
        l1_axis = np.linspace(0, len(l1_objective_values_list), len(l1_objective_values_list) + 1).astype(np.int_)
        l1_objective_values_list.insert(0, L1(weights=init).compute_output(X=None, y=None))
        l1_fig.add_trace(
            go.Scatter(x=l1_axis, y=l1_objective_values_list, mode='markers+lines', marker_color=colors[i],
                       name="eta=%.3f" % etas[i])
        )
        # L2 norm
        l2_axis = np.linspace(0, len(l2_objective_values_list), len(l2_objective_values_list) + 1).astype(np.int_)
        l2_objective_values_list.insert(0, L2(weights=init).compute_output(X=None, y=None))
        l2_fig.add_trace(
            go.Scatter(x=l2_axis, y=l2_objective_values_list, mode='markers+lines', marker_color=colors[i],
                       name="eta=%.3f" % etas[i])
        )

        # Question 4 - Printing the lowest loss achieved when running GD for L1 and L2
        print("Gradient descent using fixed LR with eta=%.3f:" % etas[i])
        print("L1's Best Solution: %s\nL1's Best Solution value: %.3f" %
              (str(l1_solution), L1(weights=l1_solution).compute_output(X=np.empty((0, 0)), y=np.empty(0))))
        print("L2's Best Solution: %s\nL2's Best Solution value: %.3f\n\n" %
              (str(l2_solution), L2(weights=l2_solution).compute_output(X=np.empty((0, 0)), y=np.empty(0))))

    l1_fig.update_layout(height=600, width=1000,
                         title_text="Convergence rates for module L1 for each learning rate",
                         xaxis_title="Iteration number",
                         yaxis_title="Objective value")
    l1_fig.show()

    l2_fig.update_layout(height=600, width=1000,
                         title_text="Convergence rates for module L2 for each learning rate",
                         xaxis_title="Iteration number",
                         yaxis_title="Objective value")
    l2_fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig_q5 = make_subplots(rows=1, cols=1)
    colors = ["blue", "orange", "green", "red"]
    for i in range(len(gammas)):
        l1_module = L1(weights=init)
        callback_func, objective_values_list, weights_list = get_gd_state_recorder_callback()
        GD = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gammas[i]), out_type='best',
                             callback=callback_func)
        solution = GD.fit(f=l1_module, X=np.empty((0, 0)), y=np.empty(0))
        print("Gradient descent for L1 using exponential LR with eta=%.3f, gamma=%.3f:\n"
              "Best Solution: %s\nObjective value for best solution: %.3f\n" %
              (eta, gammas[i], str(solution), L1(weights=solution).compute_output(X=np.empty((0, 0)), y=np.empty(0))))

        axis = np.linspace(0, len(objective_values_list), len(objective_values_list) + 1).astype(np.int_)
        objective_values_list.insert(0, L1(weights=init).compute_output(X=None, y=None))
        fig_q5.add_trace(
            go.Scatter(x=axis, y=objective_values_list, mode='markers+lines', marker_color=colors[i],
                       name="gamma=%.3f" % gammas[i])
        )

    # Plot algorithm's convergence for the different values of gamma
    fig_q5.update_layout(height=600, width=1000,
                         title_text="Convergence rates for module L1 gor each decay rate",
                         xaxis_title="Iteration number",
                         yaxis_title="Objective value")
    fig_q5.show()

    # Plot descent path for gamma=0.95
    # Descent path for L1 module
    l1_module = L1(weights=init)
    l1_callback_func, l1_objective_values_list, l1_weights_list = get_gd_state_recorder_callback()
    l1_GD = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gammas[1]),
                            out_type='best', callback=l1_callback_func)
    l1_GD.fit(f=l1_module, X=np.empty((0, 0)), y=np.empty(0))
    plot_descent_path(module=L1, descent_path=np.array(l1_weights_list),
                      title="L1 module's descent path of GD with exponential LR").show()

    # Descent path for L2 module
    l2_module = L2(weights=init)
    l2_callback_func, l2_objective_values_list, l2_weights_list = get_gd_state_recorder_callback()
    l2_GD = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gammas[1]),
                            out_type='best', callback=l2_callback_func)
    l2_GD.fit(f=l2_module, X=np.empty((0, 0)), y=np.empty(0))
    plot_descent_path(module=L2, descent_path=np.array(l2_weights_list),
                      title="L2 module's descent path of GD with exponential LR").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Question 8 - Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X=X_train, y=y_train)
    y_prob = logistic_regression.predict_proba(X=X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color="blue",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))
    ).show()

    # Question 9 - Get the best threshold for logistic regression and find it's test error
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    logistic_regression = LogisticRegression(alpha=best_threshold)
    logistic_regression.fit(X=X_train, y=y_train)
    test_error = logistic_regression.loss(X=X_test, y=y_test)
    print("Best threshold: %.3f\nTest error for LR with best threshold: %.3f\n" % (best_threshold, test_error))

    # Questions 10-11: Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    penalties = ["l1", "l2"]
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in penalties:
        validation_errors = np.zeros(len(lambdas), dtype=float)
        for i in range(len(lambdas)):
            gradient_descent = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
            logistic_regression = LogisticRegression(solver=gradient_descent, penalty=penalty, lam=lambdas[i])
            validation_errors[i] = cross_validate(logistic_regression, X_train, y_train, misclassification_error)[1]

        # Get the lambda value with the best validation error
        best_lam = lambdas[np.argmin(validation_errors)]
        # Fitting logistic regression model configured with best lambda
        logistic_regression = LogisticRegression(solver=gradient_descent, penalty=penalty, lam=best_lam)
        logistic_regression.fit(X=X_train, y=y_train)
        test_error = logistic_regression.loss(X=X_test, y=y_test)
        print("Logistic regression with %s regulation term:\nBest lambda value: %.3f\n"
              "Test error for best lambda: %.3f\n" % (penalty, best_lam, test_error))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
