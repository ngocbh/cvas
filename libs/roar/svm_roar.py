import numpy as np

from sklearn.utils import check_random_state

from libs.roar.linear_roar import LinearROAR
from libs.explainers.svm import SVMExplainer 

from utils.visualization import visualize_explanations

def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    ec = params['config']
    cat_indices = params['cat_indices']

    perturb_radius = params['perturb_radius']
    delta_max = ec.roar_params['delta_max']

    explainer = SVMExplainer(
        train_data, model.predict_proba, random_state=rng)

    w, b = explainer.explain_instance(x0,
                                      perturb_radius=perturb_radius * ec.max_distance,
                                      num_samples=ec.num_samples)

    # print("w, b: ", w, b)
    arg = LinearROAR(train_data, w, b, cat_indices, lambd=0.1, dist_type=1,
                     lr=0.01, delta_max=delta_max, max_iter=1000)
    x_ar = arg.fit_instance(x0, verbose=False)
    report = dict(feasible=arg.feasible)

    return x_ar, report
