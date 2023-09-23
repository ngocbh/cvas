import cvxpy as cp
import numpy as np
import math

from sklearn.linear_model import LogisticRegression
from scipy.special import lambertw
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.kmedians import kmedians

from utils.funcs import sqrtm_psd, lp_dist


def mpm(mu_neg, Sigma_neg, mu_pos, Sigma_pos, verbose=False):
    """mpm.

    Parameters
    ----------
    mu_neg :
        mu_neg
    Sigma_neg :
        Sigma_neg
    mu_pos :
        mu_pos
    Sigma_pos :
        Sigma_pos
    verbose :
        verbose
    """
    mu_neg = mu_neg.reshape(-1, 1)
    mu_pos = mu_pos.reshape(-1, 1)
    Sigma_neg_sqrt = sqrtm_psd(Sigma_neg)
    Sigma_pos_sqrt = sqrtm_psd(Sigma_pos)

    d = mu_neg.shape[0]

    w = cp.Variable(d)
    z = cp.Variable(2, nonneg=True)

    constraints = [
        - w.T @ mu_neg + w.T @ mu_pos == np.ones((1, 1)),
        cp.SOC(z[0], Sigma_neg_sqrt @ w),
        cp.SOC(z[1], Sigma_pos_sqrt @ w)
    ]

    objective = cp.Minimize(np.ones((1, 2)) @ z)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=verbose)

    w_opt = w.value

    kappa = 1 / prob.value
    b_opt = w_opt.T @ mu_pos - kappa * np.sqrt(w_opt.T @ Sigma_pos @ w_opt)
    b_opt = b_opt.squeeze()

    if verbose:
        print("Solving with", "="*10)
        print("mu_0: \n", mu_neg)
        print("Sigma_0: \n", Sigma_neg)
        print("mu_1: \n", mu_pos)
        print("Sigma_1: \n", Sigma_pos)
        print("Solution ", "="*10)
        print("Objective: ", prob.value)
        print("kappa: \n", kappa)
        print("w: \n", w.value)
        print("b: \n", b_opt)

    return w.value, -b_opt


def quad_rmpm(mu_neg, Sigma_neg, rho_neg, mu_pos, Sigma_pos, rho_pos, verbose=False):
    """quad_rmpm.
        Robust mimimax probability machine with quadratic divergence

    Parameters
    ----------
    mu_neg :
        mu_neg
    Sigma_neg :
        Sigma_neg
    rho_neg :
        rho_neg
    mu_pos :
        mu_pos
    Sigma_pos :
        Sigma_pos
    rho_pos :
        rho_pos
    verbose :
        verbose
    """
    d = mu_neg.shape[0]
    Sigma_neg = Sigma_neg + np.sqrt(rho_neg) * np.identity(d)
    Sigma_pos = Sigma_pos + np.sqrt(rho_pos) * np.identity(d)
    return mpm(mu_neg, Sigma_neg, mu_pos, Sigma_pos, verbose)


def bw_rmpm(mu_neg, Sigma_neg, rho_neg, mu_pos, Sigma_pos, rho_pos, verbose=False):
    mu_neg = mu_neg.reshape(-1, 1)
    mu_pos = mu_pos.reshape(-1, 1)
    Sigma_neg_sqrt = sqrtm_psd(Sigma_neg)
    Sigma_pos_sqrt = sqrtm_psd(Sigma_pos)

    d = mu_neg.shape[0]

    w = cp.Variable(d)
    z = cp.Variable(2, nonneg=True)
    t = cp.Variable(nonneg=True)

    constraints = [
        - w.T @ mu_neg + w.T @ mu_pos == np.ones((1, 1)),
        cp.SOC(z[0], Sigma_neg_sqrt @ w),
        cp.SOC(z[1], Sigma_pos_sqrt @ w),
        cp.SOC(t, w)
    ]

    objective = cp.Minimize(np.ones((1, 2)) @ z + (rho_neg + rho_pos) * t)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=verbose)

    w_opt = w.value

    kappa = 1 / prob.value
    tau_bw_pos = rho_pos * np.linalg.norm(w_opt) + np.sqrt(w_opt.T @ Sigma_pos @ w_opt)
    b_opt = w_opt.T @ mu_pos - kappa * tau_bw_pos
    # print(kappa, tau_bw_pos, kappa * tau_bw_pos)

    # tau_bw_neg = rho_neg * np.linalg.norm(w_opt) + np.sqrt(w_opt.T @ Sigma_neg @ w_opt)
    # b_opt = w_opt.T @ mu_neg + kappa * tau_bw_neg
    b_opt = b_opt.squeeze()

    if verbose:
        print("Solving with", "="*10)
        print("mu_0: \n", mu_neg)
        print("Sigma_0: \n", Sigma_neg)
        print("mu_1: \n", mu_pos)
        print("Sigma_1: \n", Sigma_pos)
        print("Solution ", "="*10)
        print("Objective: ", prob.value)
        print("kappa: \n", kappa)
        print("w: \n", w.value)
        print("b: \n", b_opt)

    return w.value, -b_opt


def fr_rmpm(mu_neg, Sigma_neg, rho_neg, mu_pos, Sigma_pos, rho_pos, verbose=False):
    mu_neg = mu_neg.reshape(-1, 1)
    mu_pos = mu_pos.reshape(-1, 1)
    Sigma_neg_sqrt = sqrtm_psd(Sigma_neg)
    Sigma_pos_sqrt = sqrtm_psd(Sigma_pos)

    d = mu_neg.shape[0]

    w = cp.Variable(d)
    z = cp.Variable(2, nonneg=True)

    constraints = [
        - w.T @ mu_neg + w.T @ mu_pos == np.ones((1, 1)),
        cp.SOC(z[0], Sigma_neg_sqrt @ w),
        cp.SOC(z[1], Sigma_pos_sqrt @ w),
    ]

    f = np.array([np.exp(rho_neg/2), np.exp(rho_pos/2)]).reshape(-1, 1)
    objective = cp.Minimize(f.T @ z)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=verbose)

    w_opt = w.value

    kappa = 1 / prob.value
    tau_fr_pos = f[1] * np.sqrt(w_opt.T @ Sigma_pos @ w_opt)
    b_opt = w_opt.T @ mu_pos - kappa * tau_fr_pos
    b_opt = b_opt.squeeze()

    if verbose:
        print("Solving with", "="*10)
        print("mu_0: \n", mu_neg)
        print("Sigma_0: \n", Sigma_neg)
        print("mu_1: \n", mu_pos)
        print("Sigma_1: \n", Sigma_pos)
        print("f:", f)
        print("Solution ", "="*10)
        print("Objective: ", prob.value)
        print("kappa: \n", kappa)
        print("w: \n", w.value)
        print("b: \n", b_opt)

    return w.value, -b_opt


def logdet_rmpm(mu_neg, Sigma_neg, rho_neg, mu_pos, Sigma_pos, rho_pos, verbose=False):
    mu_neg = mu_neg.reshape(-1, 1)
    mu_pos = mu_pos.reshape(-1, 1)
    Sigma_neg_sqrt = sqrtm_psd(Sigma_neg)
    Sigma_pos_sqrt = sqrtm_psd(Sigma_pos)

    d = mu_neg.shape[0]

    w = cp.Variable(d)
    z = cp.Variable(2, nonneg=True)

    constraints = [
        - w.T @ mu_neg + w.T @ mu_pos == np.ones((1, 1)),
        cp.SOC(z[0], Sigma_neg_sqrt @ w),
        cp.SOC(z[1], Sigma_pos_sqrt @ w),
    ]

    c = np.array([-lambertw(-np.exp(-rho_neg-1), k=-1),
                  -lambertw(-np.exp(-rho_pos-1), k=-1)]).reshape(-1, 1)
    f = np.sqrt(np.real(c))

    objective = cp.Minimize(f.T @ z)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=verbose)

    w_opt = w.value

    kappa = 1 / prob.value
    tau_logdet_pos = f[1] * np.sqrt(w_opt.T @ Sigma_pos @ w_opt)
    b_opt = w_opt.T @ mu_pos - kappa * tau_logdet_pos
    b_opt = b_opt.squeeze()

    if verbose:
        print("Solving with", "="*10)
        print("mu_0: \n", mu_neg)
        print("Sigma_0: \n", Sigma_neg)
        print("mu_1: \n", mu_pos)
        print("Sigma_1: \n", Sigma_pos)
        print("Solution ", "="*10)
        print("Objective: ", prob.value)
        print("kappa: \n", kappa)
        print("w: \n", w.value)
        print("b: \n", b_opt)

    return w.value, -b_opt


method_map = {
    'mpm': mpm,
    'quad_rmpm': quad_rmpm,
    'bw_rmpm': bw_rmpm,
    'fr_rmpm': fr_rmpm,
    'logdet_rmpm': logdet_rmpm
}


class RMPM():
    def __init__(self, method='bw_rmpm', rho_neg=0.1, rho_pos=0.1, verbose=False):
        self.method = method
        self.verbose = verbose
        self.mean_neg = None
        self.cov_neg = None
        self.mean_pos = None
        self.cov_pos = None
        self.intercept_ = None
        self.coef_ = None
        self.rho_neg = rho_neg
        self.rho_pos = rho_pos
        self.clf = method_map[method]

    def fit(self, X, y, mean_neg=None, mean_pos=None):
        # d = np.linalg.norm((X - self.org_inp), ord=2, axis=1)
        # order = np.argsort(d)
        # X_neg = X[order[y[order]==0]][:200]
        # X_pos = X[order[y[order]==1]][:200]
        X_neg = X[y == 0]
        X_pos = X[y == 1]
        if len(X_neg) <= 1 or len(X_pos) <= 1:
            print("not enough instances")
            return self

        self.mean_neg = mean_neg if mean_neg is not None else np.mean(X_neg, axis=0)
        self.cov_neg = np.cov(X_neg.T)
        self.mean_pos = mean_pos if mean_pos is not None else np.mean(X_pos, axis=0)
        self.cov_pos = np.cov(X_pos.T)


        # kmedians_instance = kmedians(X_neg, [self.mean_neg])
        # kmedians_instance.process()
        # self.mean_neg = np.array(kmedians_instance.get_medians()[0])
        # print(self.mean_neg)

        # kmedians_instance = kmedians(X_pos, [self.mean_pos])
        # kmedians_instance.process()
        # self.mean_pos = np.array(kmedians_instance.get_medians()[0])

        if self.method == 'mpm':
            w, b = self.clf(self.mean_neg, self.cov_neg,
                            self.mean_pos, self.cov_pos, verbose=self.verbose)
        else:
            w, b = self.clf(self.mean_neg, self.cov_neg, self.rho_neg,
                            self.mean_pos, self.cov_pos, self.rho_pos, verbose=self.verbose)

        self.intercept_ = b
        self.coef_ = w
        # print("rho_neg: %d, rho_pos %d, w: %s" % (self.rho_neg, self.rho_pos, w))
        # print("lim w*: ", (- self.mean_neg + self.mean_pos) / np.linalg.norm(- self.mean_neg + self.mean_pos) ** 2)
        return self
