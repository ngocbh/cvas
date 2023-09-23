import numpy as np

from sklearn.utils import check_random_state

from libs.explainers.lime_wrapper import LimeWrapper
from libs.projection.linear_proj import l2_project, LinearProj

from utils.visualization import visualize_explanations


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    ec = params['config']
    cat_indices = params['cat_indices']
    delta_max = ec.roar_params['delta_max']

    explainer = LimeWrapper(train_data, class_names=['0', '1'],
                            discretize_continuous=False, random_state=rng)

    w, b = explainer.explain_instance(x0, model.predict_proba,
                                      num_samples=ec.num_samples)

    arg = LinearProj(w, b, cat_indices, max_iter=1000)
    x_ar = arg.fit_instance(x0, delta_max=delta_max)

    # visualize_explanations(model, lines=[(w, b)], x_test=x0, mean_pos=x_ar,
                           # xlim=(-2, 4), ylim=(-4, 7), save=True)
    # raise ValueError
    report = dict(feasible=True)

    return x_ar, report

