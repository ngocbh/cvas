import os
import numpy as np
import pandas as pd
import joblib
import torch
import sklearn
import copy

from collections import defaultdict, namedtuple
from joblib import parallel_backend
from joblib.externals.loky import set_loky_pickler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler

from utils import helpers
from utils.transformer import get_transformer
from utils.funcs import compute_max_distance, lp_dist

from expt.common import synthetic_params, clf_map, method_map
from expt.common import _run_single_instance, to_mean_std
from expt.common import load_models, enrich_training_data
from expt.common import method_name_map, dataset_name_map, metric_order
from expt.expt_config import Expt3


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger):
    print("Running dataset: %s, classifier: %s, method: %s..."
               % (dname, cname, mname))
    df, _ = helpers.get_dataset(dname, params=synthetic_params)
    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    transformer = get_transformer(dname)
    X = transformer.transform(X_df)
    cat_indices = transformer.cat_indices

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)
    enriched_data = enrich_training_data(5000, X_train, cat_indices, seed)
    # ec.max_distance = compute_max_distance(X_train)

    new_config = Expt3(ec.to_dict())
    # if 'roar' in mname:
        # new_config.rmpm_params['rho_neg'] = 1.0
    new_config.max_distance = compute_max_distance(X_train)

    d = X.shape[1]
    clf = clf_map[cname]
    cur_models = load_models(dname, cname, ec.kfold, wdir)
    method = method_map[mname]

    kf = KFold(n_splits=ec.kfold)

    l1_cost = []
    cur_vald = []
    fut_vald = []
    feasible = []

    for i, (train_index, cross_index) in enumerate(kf.split(X_train)):
        X_training, X_cross = X_train[train_index], X_train[cross_index]
        y_training, y_cross = y_train[train_index], y_train[cross_index]

        model = cur_models[i]
        shifted_models = load_models(dname + f'_shift_{i}', cname, ec.num_future, wdir)

        X_all = np.vstack([X_test, X_training])
        y_all = np.concatenate([y_test, y_training])
        y_pred = model.predict(X_all)
        uds_X, uds_y = X_all[y_pred == 0], y_all[y_pred == 0]
        uds_X, uds_y = uds_X[:ec.max_ins], uds_y[:ec.max_ins]
        params = dict(train_data=X_training,
                      enriched_data=enriched_data,
                      cat_indices=cat_indices,
                      config=new_config,
                      method_name=mname,
                      dataset_name=dname)

        params['perturb_radius'] = ec.perturb_radius[dname]

        jobs_args = []

        for idx, x0 in enumerate(uds_X):
            jobs_args.append((idx, method, x0, model, shifted_models, seed, logger, params))

        rets = joblib.Parallel(n_jobs=min(num_proc, 8), prefer="threads")(joblib.delayed(_run_single_instance)(
            *jobs_args[i]) for i in range(len(jobs_args)))

        # rets = []
        # for idx, x0 in enumerate(uds_X):
            # ret = _run_single_instance(idx, method, x0, model, shifted_models, seed, logger, params)
            # rets.append(ret)

        l1_cost_ = []
        cur_vald_ = []
        fut_vald_ = []
        feasible_ = []

        for ret in rets:
            l1_cost_.append(ret.l1_cost)
            cur_vald_.append(ret.cur_vald)
            fut_vald_.append(ret.fut_vald)
            feasible_.append(ret.feasible)

        l1_cost.append(l1_cost_)
        cur_vald.append(cur_vald_)
        fut_vald.append(fut_vald_)
        feasible.append(feasible_)

    def to_numpy_array(lst):
        pad = len(max(lst, key=len))
        return np.array([i + [0]*(pad-len(i)) for i in lst])

    l1_cost = to_numpy_array(l1_cost)
    cur_vald = to_numpy_array(cur_vald)
    fut_vald = to_numpy_array(fut_vald)
    feasible = to_numpy_array(feasible)

    helpers.pdump((l1_cost, cur_vald, fut_vald, feasible),
                  f'{cname}_{dname}_{mname}.pickle', wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)

def plot_3(ec, wdir, cname, datasets, methods):
    res = defaultdict(list)
    res2 = defaultdict(list)

    for mname in methods:
        res2['method'].append(method_name_map[mname])

    for i, dname in enumerate(datasets):
        res['dataset'].extend([dataset_name_map[dname]] + [""] * (len(methods) - 1))

        joint_feasible = None
        for mname in methods:
            _, _, _, feasible = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            if joint_feasible is None:
                joint_feasible = np.ones_like(feasible)
            if '_ar' in mname:
                joint_feasible = np.logical_and(joint_feasible, feasible)
            # print(mname, np.sum(feasible, axis=1))

        temp = defaultdict(dict)

        for metric, order in metric_order.items():
            temp[metric]['best'] = -np.inf
            temp[metric]['rank'] = []

        f_feasible = (np.sum(joint_feasible, axis=1) > 0)

        for mname in methods:
            l1_cost, cur_vald, fut_vald, feasible = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            avg = {}
            avg['cost'] = np.sum(l1_cost * joint_feasible, axis=1) / np.sum(joint_feasible, axis=1)
            avg['cur-vald'] = np.sum(cur_vald * joint_feasible, axis=1) / np.sum(joint_feasible, axis=1)
            avg['fut-vald'] = np.sum(fut_vald * joint_feasible, axis=1) / np.sum(joint_feasible, axis=1)
            # avg['cost'] = np.mean(l1_cost, axis=1) 
            # avg['cur-vald'] = np.mean(cur_vald, axis=1) 
            # avg['fut-vald'] = np.mean(fut_vald, axis=1)

            for metric, order in metric_order.items():
                m, s = np.mean(avg[metric][f_feasible]), np.std(avg[metric][f_feasible])
                temp[metric][mname] = (m, s)
                temp[metric]['best'] = max(temp[metric]['best'], m * order)
                temp[metric]['rank'].append(m * order)

            temp['feasible'][mname] = np.mean(feasible)

        for mname in methods:
            res['method'].append(method_name_map[mname])
            for metric, order in metric_order.items():
                m, s = temp[metric][mname]
                is_best = (temp[metric]['best'] == m * order)
                rank = sorted(temp[metric]['rank'], reverse=True).index(m * order)
                res[metric].append(to_mean_std(m, s, rank))
                res2[f"{metric}-{dname[:2]}"].append(to_mean_std(m, s, rank))

            res[f'feasible'].append("{:.2f}".format(temp['feasible'][mname]))
            res[f'joint_feasible'].append(np.mean(joint_feasible))
            res[f'num_ins'].append(joint_feasible.shape[1])

    df = pd.DataFrame(res)
    print(df)
    filepath = os.path.join(wdir, f"{cname}{'_ar' if '_ar' in methods[0] else ''}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')

    df = pd.DataFrame(res2)
    filepath = os.path.join(wdir, f"{cname}_hor{'_ar' if '_ar' in methods[0] else ''}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')


def run_expt_3(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None, rerun=True):
    logger.info("Running ept 3...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e3.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e3.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e3.all_methods

    jobs_args = []
    if not plot_only:
        for cname in classifiers:
            cmethods = copy.deepcopy(methods)
            if cname == 'rf' and 'wachter' in cmethods:
                cmethods.remove('wachter')            

            for dname in datasets:
                for mname in cmethods:
                    filepath = os.path.join(wdir, f"{cname}_{dname}_{mname}.pickle")
                    if not os.path.exists(filepath) or rerun:
                        jobs_args.append((ec.e3, wdir, dname, cname, mname,
                            num_proc, seed, logger))

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(run)(
            *jobs_args[i]) for i in range(len(jobs_args)))

    for cname in classifiers:
        cmethods = copy.deepcopy(methods)
        if cname == 'rf' and 'wachter' in cmethods:
            cmethods.remove('wachter')            

        plot_3(ec.e3, wdir, cname, datasets, cmethods)

    logger.info("Done ept 3.")
