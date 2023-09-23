import numpy as np
import copy
import os
import torch
import joblib
import sklearn
from functools import partialmethod
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from collections import defaultdict, namedtuple

from utils import helpers
from utils.funcs import compute_max_distance, lp_dist

from classifiers import mlp, random_forest

from libs.ar import lime_ar, svm_ar, clime_ar, limels_ar
from libs.roar import lime_roar, clime_roar, limels_roar, svm_roar
from libs.wachter import wachter
from libs.projection import lime_proj, svm_proj, lime_robust_proj
from libs.rbr import rbr
from libs.dirrac import dirrac
from rmpm import rmpm_ar, rmpm_proj, rmpm_roar, rmpm_wachter, rmpm_dirrac


Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible", "runtime"])


def to_numpy_array(lst):
    pad = len(max(lst, key=len))
    return np.array([i + [0]*(pad-len(i)) for i in lst])

def load_models(dname, cname, n, wdir):
    pdir = os.path.dirname(wdir)
    pdir = os.path.join(pdir, 'checkpoints')
    models = helpers.pload(f"{cname}_{dname}_{n}.pickle", pdir)
    return models


def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.predict(x)
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds)


def enrich_training_data(num_samples, train_data, cat_indices, rng):
    rng = check_random_state(rng)
    cur_n, d = train_data.shape
    if cur_n >= num_samples:
        slt_idx = rng.choice(cur_n, num_samples, replace=False)
        return train_data[slt_idx, :]

    min_f_val = np.min(train_data, axis=0)
    max_f_val = np.max(train_data, axis=0)
    new_data = rng.uniform(min_f_val, max_f_val, (num_samples - cur_n, d))

    # new_data = rng.normal(0, 1, (num_samples - cur_n, d))
    # scaler = StandardScaler()
    # scaler.fit(train_data)
    # new_data = new_data * scaler.scale_ + scaler.mean_

    new_data[:, cat_indices] = new_data[:, cat_indices] >= 0.5

    new_data = np.vstack([train_data, new_data])
    return new_data


def to_mean_std(m, s, rank):
    if rank == 0:
        return "\\textbf{" + "{:.2f}".format(m) + "}" + " $\pm$ {:.2f}".format(s)
    if rank == 1:
        return "\\underline{" + "{:.2f}".format(m) + "}" + " $\pm$ {:.2f}".format(s)
    else:
        return "{:.2f} $\pm$ {:.2f}".format(m, s)


def _run_single_instance(idx, method, x0, model, shifted_models, seed, logger, params=dict()):
    # logger.info("Generating recourse for instance : %d", idx)

    torch.manual_seed(seed+2)
    np.random.seed(seed+1)
    random_state = check_random_state(seed)

    x_ar, report = method.generate_recourse(x0, model, random_state, params)

    l1_cost = lp_dist(x0, x_ar, p=1)
    cur_vald = model.predict(x_ar)
    fut_vald = calc_future_validity(x_ar, shifted_models)
    # print(l1_cost, cur_vald, fut_vald, report['feasible'])
    # raise ValueError

    return Results(l1_cost, cur_vald, fut_vald,
                   report['feasible'],
                   report['runtime'] if 'runtime' in report else 0)


method_name_map = {
    "lime_ar": "LIME-AR",
    "mpm_ar": "CVAS-AR",
    "mpm_threshold": "CVAS-THLD-PROJ",
    "clime_ar": "CLIME-AR",
    "limels_ar": "LIMELS-AR",
    "quad_rmpm_ar": "QUAD-CVAS-AR",
    "bw_rmpm_ar": "BW-CVAS-AR",
    "fr_rmpm_ar": "FR-CVAS-AR",
    "svm_ar": "SVM-AR",
    "lime_roar": "LIME-ROAR",
    "clime_roar": "CLIME-ROAR",
    "limels_roar": "LIMELS-ROAR",
    "svm_roar": "SVM-ROAR",
    "wachter": "Wachter",
    "wachter_threshold": "Wachter-THLD",
    "lime_proj": "LIME-PROJ",
    "lime_threshold": "LIME-THLD-PROJ",
    "lime_robust_proj": "LIME-ROBUST-PROJ",
    "mpm_proj": "CVAS-PROJ",
    "svm_proj": "SVM-PROJ",
    "fr_rmpm_proj": "FR-CVAS-PROJ",
    "quad_rmpm_proj": "QUAD-CVAS-PROJ",
    "bw_rmpm_proj": "BW-CVAS-PROJ",
    "mpm_roar": "CVAS-ROAR",
    "quad_rmpm_roar": "QUAD-CVAS-ROAR",
    "bw_rmpm_roar": "BW-CVAS-ROAR",
    'fr_rmpm_roar': "FR-CVAS-ROAR",
    'fr_rmpm_wachter': "FR-CVAS-WACHTER",
    'fr_rmpm_roar_rho': "FR-CVAS-ROAR (1)",
    'fr_rmpm_roar_delta': "FR-CVAS-ROAR (2)",
    "rbr": "RBR",
    "lime_dirrac": "LIME-DiRRAc",
    "mpm_dirrac": "CVAS-DiRRAc",
}


dataset_name_map = {
    "synthesis": "Synthetic data",
    "german": "German",
    "sba": "SBA",
    "bank": "Bank",
    "student": "Student",
    "adult": "Adult",
    "twc": "Taiwanese",
    "gmc": "GMC",
    "heloc": "HELOC",
}

metric_order = {'cost': -1, 'cur-vald': 1, 'fut-vald': 1}


method_map = {
    "lime_ar": lime_ar,
    "mpm_ar": rmpm_ar,
    "svm_ar": svm_ar,
    "clime_ar": clime_ar,
    "limels_ar": limels_ar,
    "quad_rmpm_ar": rmpm_ar,
    "bw_rmpm_ar": rmpm_ar,
    "fr_rmpm_ar": rmpm_ar,
    "wachter": wachter,
    "wachter_threshold": wachter,
    "lime_proj": lime_proj,
    "lime_threshold": lime_proj,
    "lime_robust_proj": lime_robust_proj,
    "mpm_proj": rmpm_proj,
    "mpm_threshold": rmpm_proj,
    "svm_proj": svm_proj,
    "quad_rmpm_proj": rmpm_proj,
    "bw_rmpm_proj": rmpm_proj,
    "fr_rmpm_proj": rmpm_proj,
    "lime_roar": lime_roar,
    "clime_roar": clime_roar,
    "limels_roar": limels_roar,
    "mpm_roar": rmpm_roar,
    "svm_roar": svm_roar,
    "quad_rmpm_roar": rmpm_roar,
    "bw_rmpm_roar": rmpm_roar,
    "fr_rmpm_roar": rmpm_roar,
    "fr_rmpm_wachter": rmpm_wachter,
    "rbr": rbr,
    "lime_dirrac": dirrac,
    "mpm_dirrac": rmpm_dirrac,
}




clf_map = {
    "net0": mlp.Net0,
    "mlp": mlp.Net0,
    "rf": random_forest.RandomForest,
}


train_func_map = {
    'net0': mlp.train,
    'mlp': mlp.train,
    'rf': random_forest.train,
}

# synthetic_params = dict(num_samples=50000,
#                         x_lim=(-3, 3), y_lim=(-3, 3),
#                         f=lambda x, y: y <= 1.5 * np.sin(2*x),
#                         random_state=41)

synthetic_params = dict(num_samples=1000,
                        x_lim=(-2, 4), y_lim=(-2, 7),
                        f=lambda x, y: y >= 1 + x + 2*x**2 + x**3 - x**4,
                        random_state=42)
