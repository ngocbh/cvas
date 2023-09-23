import numpy as np

from sklearn.utils import check_random_state

from libs.wachter.linear_wachter import LinearWachter
from rmpm.explainer import RMPMExplainer

from utils.visualization import visualize_explanations


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    enriched_data = params['enriched_data']
    ec = params['config']
    cat_indices = params['cat_indices']
    method_name = params['method_name'].replace('_wachter', '')
    data_name = params['dataset_name']
    perturb_radius = params['perturb_radius']
    if 'dataset_name' in params:
        dataset_name = params['dataset_name']
    else:
        dataset_name = None

    rho_neg = ec.rmpm_params['rho_neg']
    rho_pos = ec.rmpm_params['rho_pos']

    explainer = RMPMExplainer(
        train_data, model.predict_proba, random_state=random_state)

    w, b = explainer.explain_instance(x0, perturb_radius=perturb_radius * ec.max_distance,
                                      rho_neg=rho_neg, rho_pos=rho_pos,
                                      method=method_name, num_samples=ec.num_samples)

    arg = LinearWachter(w, b, cat_indices=cat_indices, y_target=1,
                  lambda_=0.4, lr=0.01, dist_type=1, max_iter=1000)
    x_ar = arg.fit_instance(x0)

    report = dict(feasible=arg.feasible)

    return x_ar, report
