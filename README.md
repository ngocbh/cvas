# Coverage-Validity-Aware Algorithmic Recourse

This repo contains source-code for the paper "Coverage-Validity-Aware Algorithmic Recourse"

## Visualization

![alt tag](https://anonymous.4open.science/api/repo/mpm-recourse/file/illus/illus.gif)

## Requirements

- Python 3.9

## Usage

1. Train a classifier

```sh
python train.py --clf mlp --data synthesis german sba student --kfold 5 --num-future 100 --num-proc 16
```

3. Run experiments

* Experiment 1 (Figure 3 and 6): Examining the fidelity and stability of local surrogate models (LIME, QUAD-MPM, BW-MPM, FR-MPM).

```sh
python run_expt.py -e 1 --datasets synthesis german sba student -clf mlp --methods lime quad_rmpm bw_rmpm fr_rmpm -uc
```

* Experiment 3 (Table 1): Generating recourses

```sh
python run_expt.py -e 3 --datasets german sba student -clf mlp -uc --methods wachter lime_proj lime_roar clime_roar limels_roar fr_rmpm_proj
```

* Experiment 3 (Table 2): Generating recourses

```sh
python run_expt.py -e 3 --datasets synthesis german sba student -clf mlp -uc --methods lime_ar clime_ar limels_ar fr_rmpm_ar
```


* Experiment 5 (Figure 4): Pareto frontier

```sh
python run_expt.py -e 5 --datasets synthesis german sba student -clf mlp -uc --methods wachter lime_roar clime_roar limels_roar fr_rmpm_proj
```
