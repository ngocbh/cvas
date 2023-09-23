import os
import numpy as np
import pandas as pd
import copy
import joblib
import torch
import sklearn
import itertools
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict, namedtuple
from joblib import parallel_backend
from joblib.externals.loky import set_loky_pickler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from utils import helpers
from utils.transformer import get_transformer
from utils.funcs import compute_max_distance, lp_dist
from utils.funcs import find_pareto

from expt.common import synthetic_params, clf_map, method_map, method_name_map
from expt.common import dataset_name_map 
from expt.common import _run_single_instance, to_numpy_array
from expt.common import load_models, enrich_training_data
from expt.expt_config import Expt5


Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible"])

param_to_vary = {
    "wachter": ["lambda"],
    "arar": ["delta_max"],
    "rbr": ["epsilon_pe", "delta_plus"],
    "lime_dirrac": ["delta_plus", "rho"],
    "mpm_dirrac": ["delta_plus", "rho"],
    "probe": ["invalidation_target"],
    "wachter_threshold": ["threshold_shift"],
    "mpm_proj": ["none"],
    "svm_proj": ["none"],
    "fr_rmpm_ar": ["rho_neg"],
    "bw_rmpm_ar": ["rho_neg"],
    "quad_rmpm_ar": ["rho_neg"],
    "fr_rmpm_proj": ["rho_neg"],
    "fr_rmpm_wachter": ["rho_neg"],
    "bw_rmpm_proj": ["rho_neg"],
    "quad_rmpm_proj": ["rho_neg"],
    "lime_roar": ["delta_max"],
    "svm_roar": ["delta_max"],
    "lime_robust_proj": ["delta_max"],
    "clime_roar": ["delta_max"],
    "limels_roar": ["delta_max"],
    "mpm_roar": ["delta_max"],
    "mpm_threshold": ["threshold_shift"],
    "lime_threshold": ["threshold_shift"],
    'fr_rmpm_roar': ["delta_max"],
    'bw_rmpm_roar': ["delta_max"],
    'quad_rmpm_roar': ["delta_max"],
}


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger, start_index=None, num_ins=None):
    # logger.info("Running dataset: %s, classifier: %s, method: %s...",
                # dname, cname, mname)
    print("Running dataset: %s, classifier: %s, method: %s..." %
                (dname, cname, mname))

    df, _ = helpers.get_dataset(dname, params=synthetic_params)
    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    transformer = get_transformer(dname)
    X = transformer.transform(X_df)
    cat_indices = transformer.cat_indices

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)
    enriched_data = enrich_training_data(1000, X_train, cat_indices, seed)

    d = X.shape[1]
    clf = clf_map[cname]
    cur_models = load_models(dname, cname, ec.kfold, wdir)

    kf = KFold(n_splits=ec.kfold)

    
    ptv_lst = param_to_vary[mname]
    param_grid = {}
    for ptv in ptv_lst:
        min_ptv = ec.params_to_vary[ptv]['min']
        max_ptv = ec.params_to_vary[ptv]['max']
        step_size = ec.params_to_vary[ptv]['step']
        values = np.arange(min_ptv, max_ptv+step_size, step_size)
        param_grid[ptv] = values

    grid = ParameterGrid(param_grid)
    print(enriched_data.shape)
    max_distance = compute_max_distance(enriched_data)
    

    mname_short = mname.replace('_delta', '').replace('_rho', '')
    method = method_map[mname_short]

    res = dict()
    res['params'] = grid
    res['delta_max_df'] = ec.roar_params['delta_max']
    res['rho_neg_df'] = ec.rmpm_params['rho_neg']
    res['cost'] = []
    res['cur_vald'] = []
    res['fut_vald'] = []
    res['feasible'] = []
    res['runtime'] = []

    for params in grid:
        # logger.info("run with params {}".format(params))
        print("run with params {}".format(params))
        # new_config = copy.deepcopy(ec)
        new_config = Expt5(ec.to_dict())
        new_config.max_distance = max_distance
        for ptv, value in params.items():
            if ptv == 'rho_neg' or ptv == 'perturb_radius':
                new_config.rmpm_params[ptv] = value
            elif ptv == 'delta_max':
                new_config.roar_params[ptv] = value
                new_config.arar_params[ptv] = value
            elif ptv == 'lambda':
                new_config.wachter_params[ptv] = value
            elif ptv == 'threshold_shift':
                new_config.threshold_shift = value
            elif ptv == 'invalidation_target':
                new_config.probe_params[ptv] = value
            elif ptv == 'epsilon_pe':
                new_config.rbr_params[ptv] = value
            elif ptv == 'epsilon_op':
                new_config.rbr_params[ptv] = value
            elif ptv == 'delta_plus':
                new_config.rbr_params[ptv] = value
                new_config.dirrac_params[ptv] = value
            elif ptv == 'rho':
                new_config.dirrac_params[ptv] = value

        train_index, _ = next(kf.split(X_train))
        X_training, y_training = X_train[train_index], y_train[train_index]

        model = cur_models[0]
        shifted_models = load_models(dname + f'_shift_{0}', cname, ec.num_future, wdir)

        X_all = np.vstack([X_test, X_training])
        y_all = np.concatenate([y_test, y_training])
        y_pred = model.predict(X_all)
        uds_X, uds_y = X_all[y_pred == 0], y_all[y_pred == 0]

        if start_index is not None or num_ins is not None:
            num_ins = num_ins or 1
            start_index = start_index or 0
            uds_X = uds_X[start_index: start_index + num_ins]
            uds_y = uds_y[start_index: start_index + num_ins]
        else:
            uds_X, uds_y = uds_X[:ec.max_ins], uds_y[:ec.max_ins]

        params = dict(train_data=X_training,
                      labels=y_training,
                      enriched_data=enriched_data,
                      cat_indices=cat_indices,
                      config=new_config,
                      method_name=mname_short,
                      dataset_name=dname)

        params['perturb_radius'] = ec.perturb_radius[dname]
        # jobs_args = []

        # for idx, x0 in enumerate(uds_X):
            # jobs_args.append((idx, method, x0, model, shifted_models, seed, logger, params))

        # rets = joblib.Parallel(n_jobs=1, prefer="threads")(joblib.delayed(_run_single_instance)(
            # *jobs_args[i]) for i in range(len(jobs_args)))
        rets = []
        for idx, x0 in enumerate(uds_X):
            ret = _run_single_instance(idx, method, x0, model, shifted_models, seed, logger, params)
            rets.append(ret)


        l1_cost = []
        cur_vald = []
        fut_vald = []
        feasible = []
        runtime = []

        for ret in rets:
            l1_cost.append(ret.l1_cost)
            cur_vald.append(ret.cur_vald)
            fut_vald.append(ret.fut_vald)
            feasible.append(ret.feasible)
            runtime.append(ret.runtime)

        res['cost'].append(np.array(l1_cost))
        res['cur_vald'].append(np.array(cur_vald))
        res['fut_vald'].append(np.array(fut_vald))
        res['feasible'].append(np.array(feasible))
        res['runtime'].append(np.array(runtime))

    helpers.pdump(res,
                  f'{cname}_{dname}_{mname}.pickle', wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)


label_map = {
    'fut_vald': "Future Validity",
    'cur_vald': "Current Validity",
    'cost': 'Cost',
}


def plot_5(ec, wdir, cname, dname, methods):
    def plot(methods, x_label, y_label, data):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots()
        marker = reversed(['*', 'v', '^', 'o', (5, 1), (5, 0), '+', 's'])
        iter_marker = itertools.cycle(marker)

        for mname in methods:
            x, y = find_pareto(data[mname][x_label], data[mname][y_label])
            ax.plot(x, y, marker=next(iter_marker),
                    label=method_name_map[mname], alpha=0.8)

        ax.set_ylabel(label_map[y_label])
        ax.set_xlabel(label_map[x_label])
        # ax.set_yscale('log')
        ax.legend(prop={'size': 14})
        filepath = os.path.join(wdir, f"{cname}_{dname}_{x_label}_{y_label}.png")
        plt.savefig(filepath, dpi=400, bbox_inches='tight')

    data = defaultdict(dict)

    for mname in methods:
        res = helpers.pload(
            f'{cname}_{dname}_{mname}.pickle', wdir)

        # print(res)
        data[mname]['params'] = res['params']
        data[mname]['rho_neg'] = res['rho_neg_df']
        data[mname]['delta_max'] = res['delta_max_df']
        data[mname]['cost'] = []
        data[mname]['fut_vald'] = []
        data[mname]['cur_vald'] = []
        data[mname]['runtime'] = []

        for i in range(len(res['params'])):
            data[mname]['cost'].append(np.mean(res['cost'][i]))
            data[mname]['fut_vald'].append(np.mean(res['fut_vald'][i]))
            data[mname]['cur_vald'].append(np.mean(res['cur_vald'][i]))
            data[mname]['runtime'].append(res['runtime'][i] if 'runtime' in res else 0)

    plot(methods, 'cost', 'fut_vald', data)
    plot(methods, 'cost', 'cur_vald', data)
    runtime_file = os.path.join(wdir, f"{cname}_{dname}_runtime.txt")
    with open(runtime_file, 'w') as f:
        for mname in methods:
            rtime = np.array(data[mname]['runtime'])
            rtime_mean = np.mean(rtime)
            rtime_std = np.std(rtime)
            f.write(f"{mname}: \n"
                    f"\t mean: {rtime_mean}\n"
                    f"\t std: {rtime_std}\n")
    for mname in methods:
        print(data[mname]['cost'])
    plt.close('all')


def plot_5_1(ec, wdir, cname, datasets, methods):

    def __plot(ax, data, dname, x_label, y_label):
        marker = reversed(['+', 'v', '^', 'o', (5, 0), '*'])
        iter_marker = itertools.cycle(marker)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors.insert(0, 'black')
        iter_color = itertools.cycle(colors)
        # num_lines = 0
        # for mname, o in data[dname].items():
        #     if param_to_vary[mname][0] != 'none':
        #         num_lines += 1

        if not any('wachter' in name for name in data[dname].keys()):
            next(iter_marker)
            next(iter_color)
        # raise ValueError

        singleton_marker = ['p', 's']
        iter_singleton_marker = itertools.cycle(singleton_marker) 
        singleton_color = ['chocolate', 'darkblue']
        iter_singleton_color = itertools.cycle(singleton_color)

        for mname, o in data[dname].items():
            if param_to_vary[mname][0] == 'none':
                ax.scatter(data[dname][mname][x_label], data[dname][mname][y_label],
                           marker=next(iter_singleton_marker), label=method_name_map[mname], alpha=0.7,
                           color=next(iter_singleton_color), zorder=10)
            else:
                x, y = find_pareto(data[dname][mname][x_label], data[dname][mname][y_label])
                ax.plot(x, y, marker=next(iter_marker),
                        label=method_name_map[mname], alpha=0.7, color=next(iter_color), zorder=10*(mname=='wachter'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(dataset_name_map[dname])
        return num_lines

    data = defaultdict(dict)
    for dname in datasets:
        for mname in methods:
            res = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)

            # print(res)
            data[dname][mname] = {}
            data[dname][mname]['params'] = res['params']
            data[dname][mname]['rho_neg'] = res['rho_neg_df']
            data[dname][mname]['delta_max'] = res['delta_max_df']
            data[dname][mname]['cost'] = []
            data[dname][mname]['fut_vald'] = []
            data[dname][mname]['cur_vald'] = []

            for i in range(len(res['params'])):
                data[dname][mname]['cost'].append(np.mean(res['cost'][i]))
                data[dname][mname]['fut_vald'].append(np.mean(res['fut_vald'][i]))
                data[dname][mname]['cur_vald'].append(np.mean(res['cur_vald'][i]))


    num_ds = len(datasets)
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams.update({'font.size': 10.5 if num_ds > 1 else 10.5})
    figsize_map = {5: (17, 5.5), 4: (12, 5.5), 3: (10, 5.5), 2: (6.5, 6.0), 1: (6, 2.5)}

    num_lines = 0

    if num_ds > 1:
        fig, axs = plt.subplots(2, num_ds, figsize=figsize_map[num_ds], constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize_map[num_ds], constrained_layout=True)

    if num_ds == 1:
        axs = axs.reshape(-1, 1)

    metrics = ['cur_vald', 'fut_vald']

    for i in range(num_ds):
        for j in range(len(metrics)):
            num_lines = __plot(axs[j, i], data, datasets[i], 'cost', metrics[j])
            if i == 0:
                axs[j, i].set_ylabel(label_map[metrics[j]])
            if j == len(metrics) - 1 or num_ds == 1:
                axs[j, i].set_xlabel(label_map['cost'])



    singleton_marker = ['p', 's']
    iter_singleton_marker = itertools.cycle(singleton_marker) 
    singleton_color = ['chocolate', 'darkblue']
    iter_singleton_color = itertools.cycle(singleton_color)
    marker = reversed(['+', 'v', '^', 'o', (5, 0), '*'])
    iter_marker = itertools.cycle(marker)
    colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors.insert(0, 'black')
    iter_color = itertools.cycle(colors)

    if not any('wachter' in name for name in data[dname].keys()):
        next(iter_marker)
        next(iter_color)

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for mname in methods:
        if param_to_vary[mname][0] == 'none':
            ax.scatter([], [], marker=next(iter_singleton_marker), label=method_name_map[mname], alpha=0.7,
                       color=next(iter_singleton_color), zorder=10)
        else:
            ax.plot([] , marker=next(iter_marker), label=method_name_map[mname], alpha=0.7, color=next(iter_color))

    num_mt = len(methods)
    max_col = 6 if num_ds > 2 else 3
    # print(num_ds)
    if num_ds > 2:
        lgd = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.23 - .1 * (len(methods) > 6)),
                  ncol=min(len(methods), 6), frameon=False)
    elif num_ds == 2:
        lgd = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.23 - 0.05),
                  ncol=min(len(methods), max_col), frameon=False)
    else:
        lgd = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.75),
                  ncol=min(len(methods), 3), frameon=False)

    if num_ds > 1:
        plt.tight_layout()
    joint_dname = ''.join([e[:2] for e in datasets])
    filepath = os.path.join(wdir, f"{cname}_{joint_dname}.pdf")
    plt.savefig(filepath, dpi=400, bbox_inches='tight')
    # plt.show()


            
def run_expt_5(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None,
               start_index=None, num_ins=None, rerun=True):
    logger.info("Running ept 5...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e5.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e5.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e5.all_methods

    if not plot_only:
        jobs_args = []

        for cname in classifiers:
            cmethods = copy.deepcopy(methods)
            if cname == 'rf' and 'wachter' in cmethods:
                cmethods.remove('wachter')            

            for dname in datasets:
                for mname in cmethods:
                    filepath = os.path.join(wdir, f"{cname}_{dname}_{mname}.pickle")
                    if not os.path.exists(filepath) or rerun:
                        jobs_args.append((ec.e5, wdir, dname, cname, mname,
                            num_proc, seed, logger, start_index, num_ins))

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(run)(
            *jobs_args[i]) for i in range(len(jobs_args)))

    for cname in classifiers:
        cmethods = copy.deepcopy(methods)
        if cname == 'rf' and 'wachter' in cmethods:
            cmethods.remove('wachter')            
        for dname in datasets:
            plot_5(ec.e5, wdir, cname, dname, cmethods)
        
        chunk_size = 4
        chunk_datasets = [datasets[i: i + chunk_size] for i in range(0, len(datasets), chunk_size)]
        for subdatasets in chunk_datasets:
            plot_5_1(ec.e5, wdir, cname, subdatasets, cmethods)

    logger.info("Done ept 5.")
