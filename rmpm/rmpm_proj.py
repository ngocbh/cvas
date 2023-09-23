import time

import numpy as np
from sklearn.utils import check_random_state

from libs.projection.linear_proj import LinearProj, l2_project
from rmpm.explainer import RMPMExplainer
from utils.visualization import visualize_explanations


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params["train_data"]
    enriched_data = params["enriched_data"]
    ec = params["config"]
    cat_indices = params["cat_indices"]
    method_name = params["method_name"].replace("_proj", "").replace("_threshold", "")
    perturb_radius = params["perturb_radius"]

    rho_neg = ec.rmpm_params["rho_neg"]
    rho_pos = ec.rmpm_params["rho_pos"]
    threshold_shift = ec.threshold_shift

    start_time = time.time()

    explainer = RMPMExplainer(
        train_data, model.predict_proba, random_state=random_state
    )

    w, b = explainer.explain_instance(
        x0,
        perturb_radius=perturb_radius * ec.max_distance,
        rho_neg=rho_neg,
        rho_pos=rho_pos,
        method=method_name,
        num_samples=ec.num_samples,
    )

    w, b = w / np.linalg.norm(w), b / np.linalg.norm(w)

    sign = -1 if np.dot(w, x0) + b <= 0 else 1
    b = b + sign * threshold_shift

    arg = LinearProj(w, b, cat_indices, max_iter=1000)
    x_ar = arg.fit_instance(x0)

    runtime = time.time() - start_time
    # visualize_explanations(model, lines=[(w, b)], x_test=x0, mean_pos=x_ar,
    # xlim=(-2, 4), ylim=(-4, 7), save=True)
    report = dict(feasible=True, runtime=runtime)

    return x_ar, report


def search_perterb_radius(model, X, y, params, logger):
    pr_lst = np.arange(0.03, 0.2, 0.02)
    logger.info("RMPM: Search best perturb_radius")
    y_pred = model.predict(X)
    uds_X = X[y_pred == 0]
    max_ins = 50
    uds_X = uds_X[:max_ins]

    logger.info("RMPM: cross_validation size: %d", len(uds_X))
    feasible = []
    best_pr = 0
    best_sum_f = 0

    for pr in pr_lst:
        logger.info("RMPM: try with perturb_radius = %.2f", pr)
        params["perturb_radius"] = pr
        sum_f = 0
        for x0 in uds_X:
            x_ar, _ = generate_recourse(x0, model, 1, params)
            sum_f += model.predict(x_ar)

        logger.info("RMPM: number of valid instances = %d", sum_f)

        if sum_f > best_sum_f:
            best_sum_f, best_pr = sum_f, pr

    logger.info("RMPM: best perturb_radius = %f", best_pr)

    return best_pr
