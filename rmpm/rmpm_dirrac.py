import numpy as np

from sklearn.utils import check_random_state

from libs.dirrac.dirrac import DRRA
from rmpm.explainer import RMPMExplainer


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)
    train_data = params["train_data"]
    # labels = params["labels"]
    # cat_indices = params['cat_indices']
    ec = params['config']
    method_name = params['method_name'].replace('_dirrac', '')
    perturb_radius = params['perturb_radius']

    rho_neg = ec.rmpm_params['rho_neg']
    rho_pos = ec.rmpm_params['rho_pos']

    ec = params["config"]
    delta_plus = ec.dirrac_params['delta_plus']
    lmbda = ec.dirrac_params['lambda']
    zeta = ec.dirrac_params['zeta']
    k, p = 1, np.array([1.0])
    rho = ec.dirrac_params['rho'] * np.ones(k)
    dim = train_data.shape[1]

    explainer = RMPMExplainer(
        train_data, model.predict_proba, random_state=random_state)

    w, b = explainer.explain_instance(x0, perturb_radius=perturb_radius * ec.max_distance,
                                      rho_neg=rho_neg, rho_pos=rho_pos,
                                      method=method_name, num_samples=ec.num_samples)

    # print("x0: ", x0)
    
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
