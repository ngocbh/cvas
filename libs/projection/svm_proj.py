import numpy as np

from sklearn.utils import check_random_state

from libs.projection.linear_proj import l2_project, LinearProj
from libs.explainers.svm import SVMExplainer 

from utils.visualization import visualize_explanations

def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    ec = params['config']
    perturb_radius = params['perturb_radius']
    cat_indices = params['cat_indices']

    explainer = SVMExplainer(
        train_data, model.predict_proba, random_state=rng)

    w, b = explainer.explain_instance(x0,
                                      perturb_radius=perturb_radius * ec.max_distance,
                                      num_samples=ec.num_samples)

    arg = LinearProj(w, b, cat_indices, max_iter=1000)
    x_ar = arg.fit_instance(x0)

    # visualize_explanations(model, lines=[(w, b)], x_test=x0, mean_pos=x_ar,
                           # xlim=(-2, 4), ylim=(-4, 7), save=True)
    report = dict(feasible=True)

    return x_ar, report
