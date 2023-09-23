import numpy as np

from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression

from utils.funcs import uniform_ball


class DBAExplainer():

    def __init__(self, train_data, predict_fn, num_cfs=5, random_state=None):
        self.random_state = check_random_state(random_state)
        self.train_data = train_data
        train_prob = predict_fn(train_data)
        self.train_label = np.argmax(train_prob, axis=1)
        self.predict_fn = predict_fn
        self.num_cfs = num_cfs

    def make_prediction(self, x):
        return np.argmax(self.predict_fn(x), axis=-1)

    def dist(self, x, y):
        return np.linalg.norm(x - y, ord=2, axis=-1)

    def find_counterfactual(self, x, k=None):
        k = k or self.num_cfs
        x_label = self.make_prediction(x)

        d = self.dist(self.train_data, x)
        order = np.argsort(d)
        x_cfs = self.train_data[order[self.train_label[order]
                                      == 1 - x_label]][:k]
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

    def explain_instance(self, x, perturb_radius=0.3, num_samples=5000):
        x_b = self.find_counterfactual(x)
        X_s = self.sample_perturbations(
            x_b, perturb_radius, num_samples, self.random_state)
        y_s = self.make_prediction(X_s)

        clf = LogisticRegression(max_iter=1000, C=1000)
        self.model = clf.fit(X_s, y_s)
        y_pred = self.model.predict(X_s)

        # return (self.model.coef_.squeeze(), self.model.intercept_), X_s, y_s, self.model
        self.data = X_s
        self.data_pred = y_s
        return self.model.coef_.squeeze(), self.model.intercept_
