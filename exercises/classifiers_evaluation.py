from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        samples, labels = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def perceptron_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(samples, labels))

        perceptron_classifier = Perceptron(callback=perceptron_callback)
        perceptron_classifier.fit(samples, labels)

        # Plot figure of loss as function of fitting iteration
        iterations_num = np.linspace(1, len(losses), len(losses)).astype(np.int_)
        losses_plot = make_subplots(rows=1, cols=1)
        losses_plot.add_trace(
            go.Scatter(x=iterations_num, y=losses, mode='lines')
        )
        losses_plot.update_layout(height=600, width=1000,
                                    title_text="Loss as function of fitting iteration for %s database" % n,
                                    xaxis_title="Iteration num",
                                    yaxis_title="Mis-classification Error (Normalized)")
        losses_plot.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples, labels = load_dataset(f)

        # Fit models and predict over training set
        GNB_classifier = GaussianNaiveBayes()
        GNB_classifier.fit(samples, labels)
        GNB_training_prediction = GNB_classifier.predict(samples)

        LDA_classifier = LDA()
        LDA_classifier.fit(samples, labels)
        LDA_training_prediction = LDA_classifier.predict(samples)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        Bayes_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Gaussian Naive Bayes - Accuracy: %.3f" % (accuracy(labels, GNB_training_prediction)),
                "Linear Discriminant Analysis - Accuracy: %.3f" % (accuracy(labels, LDA_training_prediction))
            )
        )

        # Add traces for data-points setting symbols and colors
        unique_labels = np.unique(labels)
        k = unique_labels.shape[0]
        m = samples.shape[0]
        colors = ['red', 'green', 'blue']
        shapes = ['circle', 'triangle-up', 'square']

        for i in range(m):
            for t in range(k):
                if GNB_training_prediction[i] == unique_labels[t]:
                    GNB_sample_color = colors[t]
                if LDA_training_prediction[i] == unique_labels[t]:
                    LDA_sample_color = colors[t]
                if labels[i] == unique_labels[t]:
                    sample_shape = shapes[t]

            Bayes_fig.add_trace(
                go.Scatter(mode='markers', showlegend=False, x=[samples[i, 0]], y=[samples[i, 1]],
                           marker=dict(color=GNB_sample_color, symbol=sample_shape)), row=1, col=1
            )
            Bayes_fig.add_trace(
                go.Scatter(mode='markers', showlegend=False, x=[samples[i, 0]], y=[samples[i, 1]],
                           marker=dict(color=LDA_sample_color, symbol=sample_shape)), row=1, col=2
            )

        # Add `X` dots specifying fitted Gaussians' means
        GNB_fitted_mu = GNB_classifier.mu_
        Bayes_fig.add_trace(
            go.Scatter(mode='markers', showlegend=False, x=GNB_fitted_mu[:, 0], y=GNB_fitted_mu[:, 1],
                       marker=dict(color='black', symbol='x-dot')), row=1, col=1
        )

        LDA_fitted_mu = LDA_classifier.mu_
        Bayes_fig.add_trace(
            go.Scatter(mode='markers', showlegend=False, x=LDA_fitted_mu[:, 0], y=LDA_fitted_mu[:, 1],
                       marker=dict(color='black', symbol='x-dot')), row=1, col=2
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        LDA_fitted_cov = LDA_classifier.cov_
        for t in range(k):
            # Create GBN ellipse
            GNB_fitted_class_mu = GNB_fitted_mu[t]
            GNB_fitted_class_cov = np.diag(GNB_classifier.vars_[t])
            Bayes_fig.add_trace(get_ellipse(GNB_fitted_class_mu, GNB_fitted_class_cov), row=1, col=1)

            # Create KDA ellipse
            LDA_fitted_class_mu = LDA_fitted_mu[t]
            Bayes_fig.add_trace(get_ellipse(LDA_fitted_class_mu, LDA_fitted_cov), row=1, col=2)

        # Prepare and show figure
        Bayes_fig.update_layout(
            showlegend=False, height=700, width=1400, title_x=0.5, margin=dict(t=200),
            title_font_size=30, title_text="GNB and LDA 2D Plots for dataset %s" % f
        )
        Bayes_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
