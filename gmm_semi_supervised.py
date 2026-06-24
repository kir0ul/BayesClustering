#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def e_step(U, pi_mixing_prior, mu, Sigma, seed=None, allow_singular=False):
    """
    Compute the expectation step (equation 4).

    U: unlabeled training data
    """
    N = U.shape[0]
    K = pi_mixing_prior.shape[0]
    log_probs = np.zeros((N, K))

    for n in range(N):
        for k in range(K):
            log_probs[n, k] = np.log(pi_mixing_prior[k]) + multivariate_normal.logpdf(
                x=U[n, :], mean=mu[k], cov=Sigma[k], allow_singular=allow_singular
            )
    log_norm = logsumexp(log_probs, axis=1, keepdims=True)
    gamma_resp = np.exp(log_probs - log_norm)
    return gamma_resp


def m_step(
    gamma_resp, U, X=None, Y=None, beta=0.5, reg_coef=1e-3, covariance_type=None
):
    """
    Compute the maximization step.
    """
    if X is None or Y is None:
        # Fall back to unsupervised
        beta = 0
        M = 0
    else:
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of rows"
        assert U.shape[1] == X.shape[1], "U and X must have the same number of columns"
        M, _ = X.shape
    N, D = U.shape
    K = gamma_resp.shape[1]
    eps = np.finfo(np.float32).eps

    pi_mixing_prior_new = np.zeros(K) * np.nan
    mu_new = np.zeros((K, D)) * np.nan
    Sigma_new = np.zeros((K, D, D)) * np.nan

    for k in range(K):
        if beta == 0:
            common_term = (1 - beta) * gamma_resp.sum(axis=0)[k]
        else:
            common_term = (1 - beta) * gamma_resp.sum(axis=0)[k] + beta * Y.sum(axis=0)[
                k
            ]
        if np.isnan(common_term) or common_term < eps:
            common_term = eps

        # Update mixing coefficients (equation 5)
        pi_mixing_prior_new[k] = common_term / ((1 - beta) * N + beta * M)

        # Update means (equation 6)
        gamma_resp_u = 0
        gamma_resp_x = 0
        for n in range(N):
            gamma_resp_u += gamma_resp[n, k] * U[n, :]
        if beta == 0:
            gamma_resp_x = 0
        else:
            for m in range(M):
                # gamma_resp_x += gamma_resp[m, k] * X[m, :]
                gamma_resp_x += Y[m, k] * X[m, :]
        mu_new[k, :] = ((1 - beta) * gamma_resp_u + beta * gamma_resp_x) / common_term

        # Update covariances (equation 7)
        gamma_resp_u_mu = 0
        gamma_resp_x_mu = 0
        for n in range(N):
            diff = U[n, :] - mu_new[k]
            gamma_resp_u_mu += gamma_resp[n, k] * np.outer(diff, diff)
        if beta == 0:
            gamma_resp_x_mu = 0
        else:
            for m in range(M):
                diff = X[m, :] - mu_new[k]
                gamma_resp_x_mu += (
                    # gamma_resp[m, k] * (X[m, :] - mu_new[k]) * (X[m, :] - mu_new[k]).T
                    Y[m, k] * np.outer(diff, diff)
                )
        var = ((1 - beta) * gamma_resp_u_mu) / common_term + (
            beta * gamma_resp_x_mu
        ) / common_term

        # Regularization for numerical stability (small jitter on the diagonal)
        Sigma_new[k] = var + reg_coef * np.eye(D)

        if covariance_type == "diagonal":
            # Constrain the covariance matrices to be diagonal
            Sigma_new[k] = np.diag(np.diag(Sigma_new[k]))  # + reg_coef * np.eye(D)

    return pi_mixing_prior_new, mu_new, Sigma_new


def log_likelihood(
    pi_mixing_prior, mu, Sigma, U, X=None, Y=None, beta=0.5, allow_singular=False
):
    """
    Compute the log-likelihood (equation 9).
    """
    if X is None or Y is None:
        # Fall back to unsupervised
        beta = 0
        M = 0
    else:
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of rows"
        assert U.shape[1] == X.shape[1], "U and X must have the same number of columns"
        M, _ = X.shape
    N, D = U.shape
    K = pi_mixing_prior.shape[0]
    log_probs = np.zeros((N, K))
    log_lik_unsuperv = np.zeros((N, K))
    log_lik_superv = np.zeros((M, K))

    for n in range(N):
        for k in range(K):
            log_probs[n, k] = np.log(pi_mixing_prior[k]) + multivariate_normal.logpdf(
                x=U[n, :], mean=mu[k], cov=Sigma[k], allow_singular=allow_singular
            )
    log_norm = logsumexp(log_probs, axis=1, keepdims=True)
    log_lik_unsuperv = (1 - beta) * log_norm.sum()

    if beta == 0:
        log_lik_superv = 0
    else:
        for m in range(M):
            for k in range(K):
                # ToDo: may need to convert the multiplication by `Y[m, k]` to a sum of log
                log_lik_superv[m, k] = Y[m, k] * (
                    np.log(pi_mixing_prior[k])
                    + multivariate_normal.logpdf(
                        x=X[m, :],
                        mean=mu[k],
                        cov=Sigma[k],
                        allow_singular=allow_singular,
                    )
                )
        log_lik_superv = beta * log_lik_superv.sum()

    return log_lik_unsuperv + log_lik_superv


def run_EM(
    K_components,
    U,
    X=None,
    Y=None,
    beta=0.5,
    reg_coef=1e-3,
    max_iter=300,
    tol=1e-6,
    seed=None,
    allow_singular=False,
    covariance_type=None,
):
    """
    Expectation-Maximization algorithm.

    Implemented from [1].

    [1] Yan, H.-C., Zhou, J.-H., & Pang, C. K. (2017).
    Gaussian Mixture Model Using Semisupervised Learning for Probabilistic Fault Diagnosis Under New Data Categories.
    IEEE Transactions on Instrumentation and Measurement, 66(4), 723–733.
    https://doi.org/10.1109/TIM.2017.2654552

    """
    if X is None or Y is None:
        # Fall back to unsupervised
        beta = 0

    # Initialization
    N, D = U.shape
    pi_mixing_prior = np.ones(K_components) / K_components
    # pi_mixing_prior = np.random.uniform(size=K_components)
    # pi_mixing_prior = pi_mixing_prior/pi_mixing_prior.sum()
    # mu = np.random.rand(K_components, D) * 10.0
    kmeans_init = KMeans(n_clusters=K_components, random_state=seed).fit(X=U)
    mu = np.array(kmeans_init.cluster_centers_)
    # Sigma = np.array([np.eye(D) * rng.uniform(low=0, high=0.01, size=D) for _ in range(K_components)])
    Sigma = np.array([np.eye(D) for _ in range(K_components)])

    lls = []
    for it in range(max_iter):
        # E-step
        gamma_resp = e_step(
            U=U, pi_mixing_prior=pi_mixing_prior, mu=mu, Sigma=Sigma, seed=seed
        )

        # M-step
        pi_mixing_prior, mu, Sigma = m_step(
            U=U,
            X=X,
            Y=Y,
            beta=beta,
            gamma_resp=gamma_resp,
            reg_coef=reg_coef,
            covariance_type=covariance_type,
        )

        # Log-likelihood
        ll = log_likelihood(
            U=U,
            X=X,
            Y=Y,
            beta=beta,
            pi_mixing_prior=pi_mixing_prior,
            mu=mu,
            Sigma=Sigma,
            allow_singular=allow_singular,
        )
        lls.append(ll)

        # Logging during training
        if it % 10 == 0:
            # plot_gmm(X, mu, Sigma, title=f'GMM Contours — Iteration {it+1}')
            print(f"Iteration {it:>3}/{max_iter} -- Log-likelihood: {ll}")

        # Convergence check
        if it > 0 and abs(lls[-1] - lls[-2]) < tol:
            print(f"\n--- Converged at iteration {it + 1} ---\n")
            break

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()
        ax.plot(lls, marker=".", c="k")
        ax.set_title("EM training")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-likelihood")
        # ax.set_grid(True, alpha=0.3)
        plt.show()

    labels_pred = np.argmax(gamma_resp, axis=1)
    return mu, Sigma, gamma_resp, labels_pred, pi_mixing_prior


def predict(U, pi_mixing_prior, mu, Sigma, allow_singular=False, seed=None):
    gamma_resp = e_step(
        U=U, pi_mixing_prior=pi_mixing_prior, mu=mu, Sigma=Sigma, seed=seed
    )
    uncertainty = 1 - gamma_resp.max(axis=1)
    labels_pred = np.argmax(gamma_resp, axis=1)

    N = U.shape[0]
    K = pi_mixing_prior.shape[0]
    log_probs = np.zeros((N, K))

    for n in range(N):
        for k in range(K):
            log_probs[n, k] = np.log(pi_mixing_prior[k]) + multivariate_normal.logpdf(
                x=U[n, :], mean=mu[k], cov=Sigma[k], allow_singular=allow_singular
            )
    # proba = np.exp(log_probs.sum(axis=1))
    proba = np.exp(log_probs).sum(axis=1)
    # proba = logsumexp(log_probs, axis=1)

    return gamma_resp, labels_pred, uncertainty, proba
