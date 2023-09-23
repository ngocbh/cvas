import argparse
import numpy as np

from expt.config import Config


class Expt1(Config):
    __dictpath__ = 'ec.e1'

    rmpm_params = {
        "bw_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "fr_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "quad_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
    }

    kfold = 5
    rho_neg = 0.01
    rho_pos = 0.01

    perturb_sizes = [200, 500, 1000, 2000, 5000, 10000]
    perturb_radius = 0.05
    perturb_std = 1.0
    num_samples = 5000
    max_ins = 50
    num_neighbors = 10
    sigma_neighbors = 0.001
    r_fid = 0.1
    max_distance = 1.0


class Expt2(Config):
    __dictpath__ = 'ec.e2'

    rmpm_params = {
        "bw_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "fr_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "quad_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
    }

    rho_lst_1 = np.exp(np.array([0, 2, 4, 6, 8, 10]))
    rho_lst_2 = [0, 2, 4, 6, 8, 10]

    kfold = 5
    rho_neg = 0.01
    rho_pos = 0.01

    perturb_sizes = 200
    perturb_radius = 0.05
    perturb_std = 1.0
    num_samples = 100
    max_ins = 50
    num_neighbors = 10
    sigma_neighbors = 0.001
    r_fid = 0.1
    max_distance = 1.0


class Expt3(Config):
    __dictpath__ = 'ec.e3'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime-ar', 'svm-ar']

    rmpm_params = {
        "rho_neg": 10.0,
        "rho_pos": 0.0,
    }

    wachter_params = {
        'lambda': 2.0
    }

    perturb_radius = {
        "synthesis": 0.1,
        "german": 0.2,
        "sba": 0.2,
        "student": 0.7,
        "adult": 0.2,
        "gmc": 0.2,
        "twc": 0.2,
        "heloc": 0.2,
    }

    roar_params = {
        'delta_max': 0.2,
    }

    threshold_shift = 0.0

    kfold = 5
    num_future = 100
    cross_validate = False 

    perturb_std = 1.0
    num_samples = 1000
    max_ins = 200
    sigma_neighbors = 0.001

    max_distance = 1.0


class Expt4(Config):
    __dictpath__ = 'ec.e4'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime_roar', 'fr_rmpm_ar']

    rmpm_params = {
        "rho_neg": 0.5,
        "rho_pos": 0.0,
    }


    perturb_radius = {
        "synthesis": 0.1,
        "german": 0.2,
        "sba": 0.1,
        "student": 0.7,
        "adult": 0.2,
        "gmc": 0.2,
        "twc": 0.2,
        "heloc": 0.2,
    }

    roar_params = {
        'delta_max': 0.2,
    }

    wachter_params = {
        'lambda': 0.1
    }

    probe = {
        'sigma': 0.1
    }

    params_to_vary = {
        'perturb_radius': {
            'default': 0.2,
            'min': 0.4,
            'max': 0.4,
            'step': 0.2,
        },
        'lambda': {
            'default': 0.05,
            'min': 0.05,
            'max': 5.0,
            'step': 0.5,
        },
        'rho_neg': {
            'default': 1.0,
            'min': 0.0,
            'max': 10.0,
            'step': 1.0,
        },
        'delta_max' : {
            'default': 0.05,
            'min': 0.0,
            'max': 0.2,
            'step': 0.02,
        },
        'threshold_shift' : {
            'default': 0.0,
            'min': 0.0,
            'max': 0.4,
            'step': 0.04,
        },
        'sigma' : {
            'default': 0.0,
            'min': 0.0,
            'max': 0.025,
            'step': 0.005,
        },
        'none': {
            'min': 0.0,
            'max': 0.0,
            'step': 0.1
        }
    }


    kfold = 5
    num_future = 100

    perturb_std = 1.0
    num_samples = 1000
    max_ins = 200
    max_distance = 1.0


class Expt5(Config):
    __dictpath__ = 'ec.e5'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime_roar', 'fr_rmpm_ar']

    rmpm_params = {
        "rho_neg": 0.5,
        "rho_pos": 0.0,
    }

    roar_params = {
        'delta_max': 0.2,
    }

    perturb_radius = {
        "synthesis": 0.1,
        "german": 0.2,
        "sba": 0.2,
        "student": 0.7,
        "adult": 0.2,
        "gmc": 0.2,
        "twc": 0.2,
        "heloc": 0.2,
    }

    wachter_params = {
        'lambda': 4.0,
    }

    rbr_params = {
        'delta_max': 0.5,
        'epsilon_pe': 0.0,
        'epsilon_op': 0.0,
        'sigma': 1.0,
    }

    probe_params = {
        'sigma': 0.005,
        'invalidation_target': 0.5,
    }

    arar_params = {
        'delta_max': 0.2,
    }

    dirrac_params = {
        'delta_plus': 0.5,
        'rho': 0.0,
        'lambda': 0.7,
        'zeta': 1.0,
    }

    params_to_vary = {
        'perturb_radius': {
            'default': 0.2,
            'min': 0.4,
            'max': 0.4,
            'step': 0.2,
        },
        'rho': {
            'default': 0.0,
            'min': 0.0,
            'max': 0.1,
            'step': 0.05,
        },
        'lambda': {
            'default': 0.1,
            'min': 0.05,
            'max': 5,
            'step': 0.5,
        },
        'epsilon_pe': {
            'default': 0.1,
            'min': 0.0,
            'max': 1.0,
            'step': 0.25,
        },
        'delta_plus': {
            'default': 0.5,
            'min': 0.1,
            'max': 1.1,
            'step': 0.2,
        },
        'rho_neg': {
            'default': 1.0,
            'min': 0.0,
            'max': 10.0,
            'step': 1.0,
        },
        'sigma': {
            'default': 0.0,
            'min': 0.0,
            'max': 0.025,
            'step': 0.005,
        },
        'invalidation_target': {
            'default': 0.0,
            'min': 0.1,
            'max': 1.0,
            'step': 0.1,
        },
        'delta_max': {
            'default': 0.05,
            'min': 0.0,
            'max': 0.2,
            'step': 0.02,
        },
        'threshold_shift': {
            'default': 0.0,
            'min': 0.0,
            'max': 0.4,
            'step': 0.04,
        },
        'none': {
            'min': 0.0,
            'max': 0.0,
            'step': 0.1
        }
    }

    threshold_shift = 0.0

    kfold = 5
    num_future = 100

    perturb_std = 1.0
    num_samples = 1000
    max_ins = 200
    max_distance = 1.0


class ExptConfig(Config):
    __dictpath__ = 'ec'

    e1 = Expt1()
    e2 = Expt2()
    e3 = Expt3()
    e4 = Expt4()
    e5 = Expt5()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    if args.load is not None:
        ExptConfig.from_file(args.load)
    ExptConfig.to_file(args.dump, mode=args.mode)
