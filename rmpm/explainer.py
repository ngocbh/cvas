import numpy as np

from sklearn.utils import check_random_state
from lime.lime_tabular import LimeTabularExplainer

from rmpm.classifier import RMPM
from utils.funcs import uniform_ball, compute_max_shift, distance_funcs


class RMPMExplainer():

    def __init__(self, train_data, predict_fn, num_cfs=10, random_state=None):
        self.random_state = check_random_state(random_state)
        self.train_data = train_data
        train_prob = predict_fn(train_data)
        self.train_label = np.argmax(train_prob, axis=1)
        self.predict_fn = predict_fn
        self.num_cfs = num_cfs
        self.rho_neg = None
        self.rho_pos = None

    def make_prediction(self, x):
        return np.argmax(self.predict_fn(x), axis=-1)

    def dist(self, x, y):
        return np.linalg.norm(x - y, ord=1, axis=-1)

    def find_x_boundary(self, x, k=None):
        k = k or self.num_cfs
        x_label = self.make_prediction(x)

        d = self.dist(self.train_data, x)
        order = np.argsort(d)
        x_cfs = self.train_data[order[self.train_label[order]
                                      == 1 - x_label]][:k]
        self.x_cfs = x_cfs
        best_x_b = None
        best_dist = np.inf

        for x_cf in x_cfs:
            lambd_list = np.linspace(0, 1, 100)
            for lambd in lambd_list:
                x_b = (1 - lambd) * x + lambd * x_cf
                label = self.make_prediction(x_b)
                if label == 1 - x_label:
                    dist = self.dist(x, x_b)
                    if dist < best_dist:
                        best_x_b = x_b
                        best_dist = dist
                    break
        return best_x_b

    def sample_perturbations(self, x, radius=0.3, num_samples=5000, random_state=None):
        return uniform_ball(x, radius, num_samples, random_state)

    def estimate_rho(self, x, label, radius, num_samples, method, random_state):
        covsas = []
        for _ in range(6):
            P_x = uniform_ball(x, radius, num_samples, random_state)
            y_x = self.make_prediction(P_x)
            P_x_prime = P_x[y_x == label]
            covsa = np.cov(P_x_prime.T)
            covsas.append(covsa)

        max_dist = -np.inf
        for covsa in covsas:
            max_dist = compute_max_shift(
                covsa, covsas, metric=method)
        return max_dist + 1e-3

    def explain_instance(self, x, rho_neg='auto', rho_pos='auto', method='fr_rmpm',
                         mean_neg=None, mean_pos=None,
                         perturb_radius=0.3, num_samples=5000):
        x_b = self.find_x_boundary(x)
        X_s = self.sample_perturbations(
            x_b, perturb_radius, num_samples, self.random_state)
        y_s = self.make_prediction(X_s)

        if rho_neg == 'auto':
            self.rho_neg = self.estimate_rho(
                x_b, 0, perturb_radius, num_samples, method, self.random_state)
        else:
            self.rho_neg = rho_neg
        if rho_pos == 'auto':
            self.rho_pos = self.estimate_rho(
                x_b, 1, perturb_radius, num_samples, method, self.random_state)
        else:
            self.rho_pos = rho_pos

        clf = RMPM(method, self.rho_neg, self.rho_pos)
        self.model = clf.fit(X_s, y_s, mean_neg, mean_pos)

        # return (self.model.coef_, self.model.intercept_), X_s, y_s, self.model
        self.x_b = x_b
        self.data = X_s
        self.data_pred = y_s
        return self.model.coef_, self.model.intercept_
