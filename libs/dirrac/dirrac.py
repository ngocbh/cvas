import numpy as np

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from libs.dirrac.opt import Optimization
from libs.explainers.lime_wrapper import LimeWrapper


def logistic_classifier(X, y, intercept=False):
    """ Fit the data to a logistic regression model

    Args:
        X: inputs
        y: labels (binary)

    Returns:
        coef: parameters of model
    """
    clf = LogisticRegression(fit_intercept=intercept)
    clf.fit(X, y)

    # Retrieve the model parameters
    return clf, clf.coef_


def train_theta(X, y, num_shuffle, n_components=1):
    dim = X.shape[1]
    all_coef = np.zeros((num_shuffle, dim))
    for i in range(num_shuffle):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(i + 1) * 5)
        coef = logistic_classifier(X_train, y_train)[1].T
        all_coef[i] = np.squeeze(coef)

    if n_components == 1:
        theta = np.zeros((1, dim))
        sigma = np.zeros((1, dim, dim))
        theta[0], sigma[0] = np.mean(all_coef, axis=0), np.cov(all_coef.T)

        return theta, sigma, np.array([1])

    gm = GaussianMixture(n_components=n_components, random_state=0).fit(all_coef)
    
    return gm.means_, gm.covariances_, gm.weights_


class DRRA(object):

    def __init__(self, delta_add, k, dim, p, theta, sigma, rho,
                 lmbda, zeta, dist_type='l1', real_data=False,
                 num_discrete=None, padding=False, immutable_l=None,
                 non_icr_l=None, cat_indices=0):
        self.delta_add = delta_add
        self.k = k
        self.dim = dim
        self.p = p
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lmbda = lmbda
        self.zeta = zeta
        self.dist_type = dist_type
        self.real_data = real_data
        self.padding = padding
        self.num_discrete = num_discrete
        self.cat_indices = cat_indices

        self.nm = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding, immutable_l=immutable_l, non_icr_l=non_icr_l, cat_indices=cat_indices)
        self.nwc = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, model_type='worst_case', real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding, immutable_l=immutable_l, non_icr_l=non_icr_l, cat_indices=cat_indices)
        self.gm = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, gaussian=True, real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding, immutable_l=immutable_l, non_icr_l=non_icr_l, cat_indices=cat_indices)
        self.gwc = Optimization(self.delta_add, self.k, self.dim, self.p, self.theta, self.sigma, self.rho, self.lmbda, self.zeta, self.dist_type, gaussian=True, model_type='worst_case', real_data=self.real_data, num_discrete=self.num_discrete, padding=self.padding, immutable_l=immutable_l, non_icr_l=non_icr_l, cat_indices=cat_indices)
        self.models = {'nm': self.nm, 'nwc': self.nwc, 'gm': self.gm, 'gwc': self.gwc}

    def fit_instance(self, x, model='nm'):
        """ Recorse action with an instance

        Args:
            x: original instance
            model: model type

        Returns:
            x_opt: recourse of x
        """
        x = np.concatenate([x, np.asarray([1.0])])
        out = self.models[model].recourse_action(x, 10)
        try:
            f_opt, x_opt = out
        except:
            x_opt = x.copy()

        return x_opt[:-1]

    def fit_data(self, data, model='nm'):
        counterfactual_samples = np.zeros((len(data), self.dim))
        for i in tqdm(range(len(data))):
            counterfactual_samples[i] = self.fit_instance(data[i], model=model)

        return counterfactual_samples


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)
    train_data = params["train_data"]
    # labels = params["labels"]
    cat_indices = params['cat_indices']

    ec = params["config"]
    delta_plus = ec.dirrac_params['delta_plus']
    lmbda = ec.dirrac_params['lambda']
    zeta = ec.dirrac_params['zeta']
    k, p = 1, np.array([1.0])
    rho = ec.dirrac_params['rho'] * np.ones(k)
    dim = train_data.shape[1]

    explainer = LimeWrapper(train_data, class_names=['0', '1'],
                            discretize_continuous=False, random_state=rng)

    # print("x0: ", x0)
    w, b = explainer.explain_instance(x0, model.predict_proba,
                                      num_samples=ec.num_samples)
    
    # Expand dims, parameters of original classifier
    theta = np.concatenate([w, np.asarray([b])]).reshape(1, -1)
    sigma = np.expand_dims(np.identity(dim + 1), axis=0)
    # print("w, b: ", w, b)

    # theta, sigma, p = train_theta(train_data, labels, num_shuffle=100, n_components=k)

    arg = DRRA(delta_plus, k, dim + 1, p, theta, sigma, rho, lmbda, zeta,
               dist_type='l1', real_data=False, num_discrete=None,
               padding=True, immutable_l=None, non_icr_l=None, cat_indices=None)

    x_ar = arg.fit_instance(x0, model='nm')

    # print("lambda", lmbda)
    # print("x_0: ", x0)
    # print("x_ar: ", x_ar)
    # print("logit: x_0", np.dot(x0, w) + b)
    # print("logit: x_ar", np.dot(x_ar, w) + b)
    # print("delta_plus: ", delta_plus, "dist: ", np.linalg.norm(x_ar-x0, 1),
    #       "predict_prob: ", model.predict_proba(x_ar)[1])
    # print("="*10, "\n")
    report = dict(feasible=True)
    # raise ValueError

    return x_ar, report
