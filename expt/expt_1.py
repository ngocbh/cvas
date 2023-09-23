import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import joblib
import random
import json
import torch
import itertools
from collections import defaultdict, namedtuple
from copy import deepcopy

from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from functools import partialmethod
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from utils import helpers
from utils.visualization import visualize_explanations
from expt.common import synthetic_params, clf_map, dataset_name_map
from libs.explainers.clime import CLime
from libs.explainers.limels import LimeLS

from classifiers.mlp import Net0
from libs.explainers.clime import CLime
from libs.explainers.lime_wrapper import LimeWrapper
from libs.explainers.dba import DBAExplainer
from libs.explainers.svm import SVMExplainer
from utils.funcs import compute_robustness, compute_fidelity, \
    compute_max_distance, compute_max_shift
from utils.transformer import get_transformer
from rmpm.classifier import RMPM
from utils import helpers
from utils.visualization import visualize_explanations
from rmpm.explainer import RMPMExplainer
from expt.expt_config import Expt1


Results = namedtuple("Results", ["rob", "fid", "cond_cov_pos", "cond_cov_neg", "fr_neg",
                                 "quad_pos", "bw_pos", "fr_pos", "mean_neg", "mean_pos"])
Results.__new__.__defaults__ = (0,) * len(Results._fields)


def load_models(dname, cname, n, wdir):
    pdir = os.path.dirname(wdir)
    pdir = os.path.join(pdir, 'checkpoints')
    models = helpers.pload(f"{cname}_{dname}_{n}.pickle", pdir)
    return models

def _run_lime(idx, x, model, train_data, method, ec, random_state, logger):
    logger.info("Running method %s with instance %d...", method, idx)
    explainer = LimeWrapper(train_data, class_names=['0', '1'],
                            discretize_continuous=False, random_state=random_state)

    exps = []

    for x0 in x:
        w, b = explainer.explain_instance(x0, model.predict_proba,
                                          num_samples=ec.num_samples)

        exps.append((w, b))

    seed = int(np.linalg.norm(x[0]) * 10000)
    rob = compute_robustness(exps)
    fid = compute_fidelity(
        x[0], exps[0], model.predict_proba, ec.r_fid * ec.max_distance, random_state=seed)
    logger.info("Done method %s, instance %d!", method, idx)
    return Results(rob, fid)

def _run_rmpm(idx, x, model, train_data, method, ec, random_state, logger):
    logger.info("Running method %s with instance %d...", method, idx)
    explainer = RMPMExplainer(
        train_data, model.predict_proba, random_state=random_state)

    exps = []
    mean_negs = []
    cov_negs = []
    mean_poss = []
    cov_poss = []

    cond_cov_neg_lst = []
    cond_cov_pos_lst = []

    for i, x0 in enumerate(x):
        x0_label = model.predict(x0)
        rho_neg = ec.rmpm_params[method]['rho_neg']
        rho_pos = ec.rmpm_params[method]['rho_pos']
        # rho_neg = rho_pos = 'auto'
        w, b = explainer.explain_instance(x0, perturb_radius=ec.perturb_radius * ec.max_distance,
                                          rho_neg=rho_neg, rho_pos=rho_pos,
                                          method=method, num_samples=ec.num_samples)
        logger.info("explained %d", i)

        cond_cov_neg_lst.append(np.linalg.cond(explainer.model.cov_neg))
        cond_cov_pos_lst.append(np.linalg.cond(explainer.model.cov_pos))

        exps.append((w, b))

    seed = int(np.linalg.norm(x[0]) * 10000)
    rob = compute_robustness(exps)
    fid = compute_fidelity(
        x[0], exps[0], model.predict_proba, ec.r_fid * ec.max_distance, random_state=seed)
    mean_cond_neg = np.mean(cond_cov_neg_lst)
    mean_cond_pos = np.mean(cond_cov_pos_lst)

    logger.info("Done method %s, instance %d!", method, idx)
    return Results(rob, fid, mean_cond_neg, mean_cond_pos)


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger):
    logger.info("Running dataset: %s, classifier: %s, method: %s...",
                dname, cname, mname)

    torch.manual_seed(seed+2)
    np.random.seed(seed+1)
    random_state = check_random_state(seed)

    df, _ = helpers.get_dataset(dname, params=synthetic_params)
    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    transformer = get_transformer(dname)
    X = transformer.transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)

    ec.max_distance = compute_max_distance(X_train)
    # print(ec.max_distance)

    d = X.shape[1]

    cur_models = load_models(dname, cname, ec.kfold, wdir)
    model = cur_models[0]
    method = method_map[mname]

    d = X_test.shape[1]
    res = defaultdict(dict)

    jobs_args = []
    for idx, x0 in enumerate(X_test[:ec.max_ins]):
        x_neighbors = x0 + \
            random_state.randn(ec.num_neighbors, d) * ec.sigma_neighbors
        x_neighbors = np.vstack([x0, x_neighbors])

        # run
        jobs_args.append((idx, x_neighbors, model, X_train, mname,
                          ec, random_state, logger))

    # fix epsilon, varying k
    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(method)(
        *jobs_args[i]) for i in range(len(jobs_args)))

    robustness = []
    fidelity = []
    cond_cov_neg = []
    cond_cov_pos = []

    for r in rets:
        robustness.append(r.rob)
        fidelity.append(r.fid)
        cond_cov_neg.append(r.cond_cov_neg)
        cond_cov_pos.append(r.cond_cov_pos)

    res = {
        mname: {
            'fid': np.mean(fidelity),
            'rob': np.mean(robustness),
            'cond_cov_neg': np.mean(cond_cov_neg),
            'cond_cov_pos': np.mean(cond_cov_pos),
        }
    }
    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)
    return res


method_map = {
    "lime": _run_lime,
    "bw_rmpm": _run_rmpm,
    "quad_rmpm": _run_rmpm,
    "fr_rmpm": _run_rmpm,
    "mpm": _run_rmpm,
    "logdet_rmpm": _run_rmpm,
}

def run_varying_perturb_samples(ec, wdir, dname, cname, mname,
                                num_proc, seed, logger):
    logger.info("Run experiment 1.")
    ret = dict()
    for num_samples in ec.perturb_sizes:
        ec_new = Expt1(ec.to_dict())
        ec_new.num_samples = num_samples
        res = run(ec_new, wdir, dname, cname, mname,
                  num_proc, seed, logger)
        for m, o in res.items():
            if m not in ret:
                ret[m] = defaultdict(list)
            ret[m]['fid'].append(o['fid'])
            ret[m]['rob'].append(o['rob'])
            ret[m]['cond_cov_neg'].append(o['cond_cov_neg'])
            ret[m]['cond_cov_pos'].append(o['cond_cov_pos'])

    helpers.pdump((ec_new.perturb_sizes, ret),
                  f'{cname}_{dname}_{mname}.pickle', wdir)

    logger.info("Done experiment 1.")

method_name = {
    "bw_rmpm": "BW-CVAS",
    "fr_rmpm": "FR-CVAS",
    "lime": "LIME",
    "mpm": "MPM",
    "quad_rmpm": "QUAD-CVAS",
    "logdet_rmpm": "LOGDET-CVAS",
}
metric_name = {
    "fid": "Local Fidelity",
    "rob": "Sensitivity",
    "cond_cov_neg": "Negative class",
    "cond_cov_pos": "Positive class",
}


def plot_1(ec, wdir, cname, dname, methods):
    def __plot(res, metric, ec, wdir):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots()
        marker = reversed(['+', 'v', '^', 'o', (5, 1), (5, 0)])
        iter_marker = itertools.cycle(marker)
        for mname, o in res.items():
            ax.plot(ec.perturb_sizes, o[metric], marker=next(iter_marker),
                    label=method_name[mname], alpha=0.8)

        ax.set_xticks(ec.perturb_sizes)
        ax.set_ylabel(metric_name[metric] if metric in metric_name else metric)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlabel("No. perturbation samples")
        ax.set_xscale('log')
        ax.legend(prop={'size': 14})
        filepath = os.path.join(wdir, f"{cname}_{dname}_{metric}.png")
        plt.savefig(filepath, dpi=400, bbox_inches='tight')
        # plt.show()

    res = {}
    for mname in methods:
        perturb_sizes, r = helpers.pload(
            f'{cname}_{dname}_{mname}.pickle', wdir)
        for m, o in r.items():
            res[m] = o

    __plot(res, 'fid', ec, wdir)
    __plot(res, 'rob', ec, wdir)
    __plot(res, 'cond_cov_neg', ec, wdir)
    __plot(res, 'cond_cov_pos', ec, wdir)

    pass

def plot_2(ec, wdir, cname, datasets, methods):
    def __plot(ax, perturb_sizes, data, dname, metric):
        marker = reversed(['+', 'v', '^', 'o', (5, 1), (5, 0)])
        iter_marker = itertools.cycle(marker)
        for mname, o in data[dname].items():
            ax.plot(perturb_sizes, o[metric], marker=next(iter_marker),
                    label=method_name[mname], alpha=0.7)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xscale('log')
        ax.set_title(dataset_name_map[dname])

    data = defaultdict(dict)

    perturb_sizes = None
    for dname in datasets:
        for mname in methods:
            perturb_sizes, r = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            for m, o in r.items():
                data[dname][m] = o

    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams.update({'font.size': 10.5})
    num_ds = len(datasets)
    figsize_map = {4: (12, 5.5), 3: (10, 5.5), 2: (6.5, 5.8)}
    fig, axs = plt.subplots(2, num_ds, figsize=figsize_map[num_ds])

    # metrics = list(metric_name.keys())
    metrics = ['rob', 'fid']

    for i in range(num_ds):
        for j in range(len(metrics)):
            __plot(axs[j, i], perturb_sizes, data, datasets[i], metrics[j])
            if i == 0:
                axs[j, i].set_ylabel(metric_name[metrics[j]])
            if j == len(metrics) - 1:
                axs[j, i].set_xlabel("No. samples")

    marker = reversed(['+', 'v', '^', 'o', (5, 1), (5, 0)])
    iter_marker = itertools.cycle(marker)
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for mname in methods:
        ax.plot([] , marker=next(iter_marker), label=method_name[mname], alpha=0.7)

    num_mt = len(methods)
    max_col = 6 if num_ds > 2 else 4
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.23 - .1 * (num_mt > max_col)),
              ncol=min(num_mt, max_col), frameon=False)
    plt.tight_layout()
    joint_dname = ''.join([e[:2] for e in datasets])
    filepath = os.path.join(wdir, f"{cname}_{joint_dname}.png")
    plt.savefig(filepath, dpi=400, bbox_inches='tight')


def plot_3(ec, wdir, cname, datasets, methods):
    def __plot(ax, data, dname, metrics):
        marker = reversed(['+', 'v', '^', 'o', (5, 1), (5, 0)])
        iter_marker = itertools.cycle(marker)
        for metric in metrics:
            o = data[dname]['fr_rmpm']
            ax.plot(ec.perturb_sizes, o[metric], marker=next(iter_marker),
                    label=metric_name[metric], alpha=0.7)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set_title(dataset_name_map[dname])

    data = defaultdict(dict)

    for dname in datasets:
        for mname in methods:
            perturb_sizes, r = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            for m, o in r.items():
                data[dname][m] = o

    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams.update({'font.size': 10.5})
    num_ds = len(datasets)
    figsize_map = {4: (12, 3), 3: (10, 4), 2: (6.5, 5)}
    fig, axs = plt.subplots(1, num_ds, figsize=figsize_map[num_ds])

    # metrics = list(metric_name.keys())
    metrics = ['cond_cov_neg', 'cond_cov_pos']

    for i in range(num_ds):
        __plot(axs[i], data, datasets[i], metrics)
        if i == 0:
            axs[i].set_ylabel("Condition number")
        axs[i].set_xlabel("No. samples")

    plt.tight_layout()
    joint_dname = ''.join([e[:2] for e in datasets])
    filepath = os.path.join(wdir, f"{cname}_{joint_dname}_cond.pdf")
    plt.savefig(filepath, dpi=400, bbox_inches='tight')


def run_expt_1(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None):
    logger.info("Running ept 1...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e1.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e1.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e1.all_methods

    logger.info(json.dumps(ec.e1.to_dict(), indent=4))

    for cname in classifiers:
        for dname in datasets:
            if not plot_only:
                for mname in methods:
                    run_varying_perturb_samples(ec.e1, wdir, dname, cname, mname,
                                                num_proc, seed, logger)
            plot_1(ec.e1, wdir, cname, dname, methods)

        plot_2(ec.e1, wdir, cname, datasets, methods)
        plot_3(ec.e1, wdir, cname, datasets, methods)

    logger.info("Done ept 1.")