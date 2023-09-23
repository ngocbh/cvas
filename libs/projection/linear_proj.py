from tqdm import tqdm

import cvxpy as cp
import numpy as np
import time

import torch
import torch.nn as nn

import gurobipy as grb


def l2_project(x0, w, b, epsilon=0.01):
    x_proj = x0 - min(0, np.dot(w, x0) + b - epsilon) * w / (np.linalg.norm(w) ** 2)
    return x_proj


def l1_project(x0, w, b, epsilon, cat_pos):
    n, = x0.shape
    n_cat = len(cat_pos)
    n_num = n - n_cat
    x0_num, x0_cat = x0[:n_num], x0[n_num:]
    w_num, w_cat = w[:n_num], w[n_num:]

    if n_cat > 0 and n_num > 0:
        x_num = cp.Variable(n_num)
        x_cat = cp.Variable(n_cat, boolean=True)
        # print(x0_num.shape)
        # raise ValueError

        constraints = [
            w_num.T @ x_num + w_cat.T @ x_cat + b >= epsilon
        ]

        objective = cp.Minimize(cp.norm(x0_num - x_num, 1) + cp.norm(x0_cat - x_cat, 1))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        x_num_opt = x_num.value
        x_cat_opt = x_cat.value
        # print(x_num_opt, x_cat_opt)
        x_opt = np.concatenate([x_num_opt, x_cat_opt])
    elif n_num > 0 or n_cat > 0:
        x = cp.Variable(n, boolean=(n_cat > 0))

        constraints = [
            w.T @ x + b >= epsilon
        ]

        objective = cp.Minimize(cp.norm(x0 - x, 1))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        x_opt = x.value
    return x_opt


def l1_robust_project(x0, w, b, epsilon, cat_pos, delta_max):
    n, = x0.shape
    n_cat = len(cat_pos)
    # n_cat = 1
    n_num = n - n_cat
    x0_num, x0_cat = x0[:n_num], x0[n_num:]
    w_num, w_cat = w[:n_num], w[n_num:]
    # print(cp.hstack([x0_num, x0_cat]))
    # print(delta_max)
    # print(n_cat)
    # print(np.hstack([x0_num, x0_cat]))
    # print("norm w before: ", np.linalg.norm(w, 2))
    # print(cp.norm(cp.hstack([x0_num, x0_cat]), 2))
    # raise ValueError
    b = b / np.linalg.norm(w, 2)
    w = w / np.linalg.norm(w, 2)

    if n_cat > 0 and n_num > 0:
        x_num = cp.Variable(n_num)
        x_cat = cp.Variable(n_cat, boolean=True)
        # print(x0_num.shape)
        # raise ValueError
        x = cp.hstack([x_num, x_cat])

        constraints = [
            w.T @ x + b - delta_max * cp.norm(x, 2) >= epsilon
        ]

        objective = cp.Minimize(cp.norm(x0 - x, 1))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        x_num_opt = x_num.value
        x_cat_opt = x_cat.value
        x_opt = np.concatenate([x_num_opt, x_cat_opt])
    elif n_num > 0 or n_cat > 0:
        x = cp.Variable(n, boolean=(n_cat > 0))

        constraints = [
            w.T @ x + b - delta_max * cp.norm(x, 2) >= epsilon
        ]

        objective = cp.Minimize(cp.norm(x0 - x, 1))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)

        x_opt = x.value

    return x_opt


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.copy()
    for pos in cat_pos:
        x_enc[pos] = np.clip(np.round(x_enc[pos]), 0, 1)
    return x_enc


class LinearProj(object):
    """ Class for generate counterfactual samples for framework: ROAR """
    DECISION_THRESHOLD = 0.5

    def __init__(self, coef, intercept, cat_indices=list(), lr=0.5, epsilon=0.1, max_iter=50, encoding_constraints=True):
        self.coef_ = coef
        self.intercept_ = intercept
        self.lr = lr
        self.cat_indices = cat_indices
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.encoding_constraints = encoding_constraints

    def f(self, x):
        return np.dot(x, self.coef_) + self.intercept_

    def fit_instance(self, x0, delta_max=None, verbose=False):
        x_t = x0.copy()
        cat_pos = self.cat_indices if self.encoding_constraints else []
        if delta_max is None:
            x_l1 = l1_project(x_t, self.coef_, self.intercept_, epsilon=self.epsilon, cat_pos=cat_pos)
        else:
            x_l1 = l1_robust_project(x_t, self.coef_, self.intercept_, epsilon=self.epsilon,
                                     cat_pos=cat_pos, delta_max=delta_max)
        # x_l2 = l2_project(x_t, self.coef_, self.intercept_, epsilon=self.epsilon)
        # print(x0)
        # print(np.linalg.norm(x_l1 - x0, 1), x_l1.T @ self.coef_ + self.intercept_)
        # print(x_l1)
        # print(np.linalg.norm(x_l2 - x0, 1), x_l2.T @ self.coef_ + self.intercept_)
        # print(x_l2)
        # raise ValueError

        # for it in range(self.max_iter):
            # x_t = l2_project(x_t, self.coef_, self.intercept_, epsilon=self.epsilon)
            # x_t = reconstruct_encoding_constraints(x_t, self.cat_indices)

            # if self.f(x_t) >= self.epsilon - 1e-7:
                # break

        # self.feasible = (self.f(x_t) >= self.epsilon - 1e-7)

        # print(np.linalg.norm(x_t - x0, 1), x_t.T @ self.coef_ + self.intercept_)
        # print(x_t)
        # raise ValueError
        return x_l1

