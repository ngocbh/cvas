import numpy as np

from sklearn.utils import check_random_state

from libs.roar.linear_roar import LinearROAR
from rmpm.explainer import RMPMExplainer

from utils.visualization import visualize_explanations


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    enriched_data = params['enriched_data']
    ec = params['config']
    method_name = params['method_name'].replace('_roar', '')
    data_name = params['dataset_name']
    cat_indices = params['cat_indices']
    perturb_radius = params['perturb_radius']

    rho_neg = ec.rmpm_params['rho_neg']
    rho_pos = ec.rmpm_params['rho_pos']

    delta_max = ec.roar_params['delta_max']

    explainer = RMPMExplainer(
        train_data, model.predict_proba, random_state=random_state)

    w, b = explainer.explain_instance(x0, perturb_radius=perturb_radius * ec.max_distance,
                                      rho_neg=rho_neg, rho_pos=rho_pos,
                                      method=method_name, num_samples=ec.num_samples)
    # print("w, b: ", w, b)

    arg = LinearROAR(train_data, w, b, cat_indices, lambd=0.1, dist_type=1,
                     lr=0.01, delta_max=delta_max, max_iter=1000)

    x_ar = arg.fit_instance(x0, verbose=False)

    # print("x_0: ", x0)
    # print("x_ar: ", x_ar)
    # print("logit: x_0", np.dot(x0, w) + b)
    # print("logit: x_ar", np.dot(x_ar, w) + b)
    # print("delta_max: ", delta_max, "dist: ", np.linalg.norm(x_ar-x0, 1),
          # "predict_prob: ", model.predict_proba(x_ar)[1])
    # raise ValueError
    # visualize_explanations(model, lines=[(w, b)], x_test=x0, mean_pos=x_ar,
                           # xlim=(-2, 4), ylim=(-4, 7), save=True)
    report = dict(feasible=arg.feasible)

    return x_ar, report


def search_perterb_radius(model, X, y, params, logger):
    pr_lst = np.arange(0.03, 0.2, 0.02)
    logger.info('RMPM: Search best perturb_radius')
    y_pred = model.predict(X)
    uds_X = X[y_pred == 0]
    max_ins = 50
    uds_X = uds_X[:max_ins]

    logger.info('RMPM: cross_validation size: %d', len(uds_X))
    feasible = []
    best_pr = 0
    best_sum_f = 0

    for pr in pr_lst:
        logger.info('RMPM: try with perturb_radius = %.2f', pr)
        params['perturb_radius'] = pr
        sum_f = 0
        for x0 in uds_X:
            x_ar, _ = generate_recourse(x0, model, 1, params)
            sum_f += model.predict(x_ar)

        logger.info('RMPM: number of valid instances = %d', sum_f)

        if sum_f > best_sum_f:
            best_sum_f, best_pr = sum_f, pr

    logger.info('RMPM: best perturb_radius = %f', best_pr)

    return best_pr


def search_lambda(model, X, y, params, logger):
    lbd_list = np.arange(0.01, 0.1, 0.01)
    logger.info('ROAR: Search best lambda')

    y_pred = model.predict(X)
    uds_X = X[y_pred == 0]
    max_ins = 50
    uds_X = uds_X[:max_ins]

    logger.info('ROAR: cross_validation size: %d', len(uds_X))

    feasible = []
    best_lbd = 0
    best_sum_f = 0

    for lbd in lbd_list:
        logger.info('ROAR: try with lambda = %.2f', lbd)
        params['lambda'] = lbd
        sum_f = 0
        for x0 in uds_X:
            x_ar, _ = generate_recourse(x0, model, 1, params)
            sum_f += model.predict(x_ar)

        logger.info('ROAR: number of valid instances = %d', sum_f)

        if sum_f >= best_sum_f:
            best_sum_f, best_lbd = sum_f, lbd


    logger.info('ROAR: best lambda = %f', best_lbd)
    return best_lbd
