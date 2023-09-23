from torch.autograd import Variable

from sklearn.utils import check_random_state

from libs.wachter.wachter import Wachter

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import gurobipy as grb


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc


class LinearClassifier(nn.Module):
    def __init__(self, w, b):
        self.w = torch.from_numpy(w).float()
        self.b = torch.tensor(b).float()
        super(LinearClassifier, self).__init__()

    def forward(self, x):
        return torch.sigmoid(self.w.T @ x + self.b)


class LinearWachter(object):
    """ Class for generate counterfactual samples for framework: Wachter """
    DECISION_THRESHOLD = 0.5

    def __init__(self, coef, intercept, cat_indices=list(), y_target=1, lambda_=0.1,
                 lr=0.01, dist_type=1, max_iter=1000, encoding_constraints=False):
        self.coef = coef
        self.intercept = intercept
        model = LinearClassifier(coef, intercept)
        self.wachter = Wachter(model, cat_indices, y_target, lambda_,
                               lr, dist_type, max_iter, encoding_constraints)

    def fit_instance(self, x0):
        x_r = self.wachter.fit_instance(x0)
        self.feasible = self.wachter.feasible
        return x_r


