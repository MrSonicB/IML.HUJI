import sys
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples_q1 = np.random.normal(10, 1, 1000)
    estimator_q1 = UnivariateGaussian()
    estimator_q1.fit(samples_q1)
    print((estimator_q1.mu_, estimator_q1.var_))

    # Question 2 - Empirically showing sample mean is consistent
    estimator_q2 = UnivariateGaussian()
    samples_sizes = np.linspace(10, 1000, 100).astype(np.int_)
    errors = []

    for samples_size in samples_sizes:
        samples = np.random.normal(10, 1, samples_size)
        estimator_q2.fit(samples)
        errors.append(np.abs(estimator_q2.mu_ - 10))

    fig_question2 = make_subplots(rows=1, cols=1)
    fig_question2.add_trace(
        go.Scatter(x=samples_sizes, y=errors, mode='markers+lines')
    )
    fig_question2.update_layout(height=600, width=1000,
                                title_text="Estimation Error distribution",
                                xaxis_title="Sample size",
                                yaxis_title="Different from real expectation")
    fig_question2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(samples_q1)
    pdf_values = estimator_q1.pdf(sorted_samples)
    fig_question3 = make_subplots(rows=1, cols=1)
    fig_question3.add_trace(
        go.Scatter(x=sorted_samples, y=pdf_values, mode='markers+lines')
    )
    fig_question3.update_layout(height=600, width=1000,
                                title_text="Univariate Gaussian Empirical PDF",
                                xaxis_title="Sample from q1",
                                yaxis_title="Estimator's pdf values")
    fig_question3.show()
    """
    I'm expecting to see a correlation between the sample's density and the value of the estimator's PDF.
    That is more samples will concentrate in higher values while less samples will concentrate in smaller PDF values.
    """

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    expectation_vector = np.array([0, 0, 4, 0])
    covariance_matrix = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples_q4 = np.random.multivariate_normal(expectation_vector, covariance_matrix, 1000)
    estimator_q4 = MultivariateGaussian()
    estimator_q4.fit(samples_q4)
    print(estimator_q4.mu_)
    print(estimator_q4.cov_)

    # Question 5 - Log-Likelihood evaluation
    f1_values = np.linspace(-10, 10, 200)
    f3_values = np.linspace(-10, 10, 200)
    log_likelihood_matrix = np.zeros((200, 200))

    for i in range(200):
        for j in range(200):
            tmp_mu_vector = np.array([f1_values[i], 0, f3_values[j], 0])
            log_likelihood_matrix[i][j] = MultivariateGaussian.log_likelihood(tmp_mu_vector,
                                                                              covariance_matrix, samples_q4)

    go.Figure(go.Heatmap(x=f3_values, y=f1_values, z=log_likelihood_matrix),
              layout=go.Layout(title="Multivariate gaussian log-likelihood Heatmap",
                               xaxis_title="f3 values",
                               yaxis_title="f1 values",
                               height=1000, width=1000)).show()
    """
    According to the plot, we can tell that as (f1,f3)→(0,4), our model will better predict the real distribution of the samples from question4.
    That is make sense, since for (f1,f3)= (0,4) our model will predict the
    real distribution of those samples.
    """

    # Question 6 - Maximum log-likelihood
    i, j = np.unravel_index(log_likelihood_matrix.argmax(), log_likelihood_matrix.shape)
    print("Maximum log likelihood:\n"
          "f1={:.3f}\nf3={:.3f}\nlog likelihood value={:.3f}".format(f1_values[i], f3_values[j], log_likelihood_matrix[i][j]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
