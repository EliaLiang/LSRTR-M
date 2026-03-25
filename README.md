# LSRTR-M

Code and documentation by:

Xiao Liang  
Department of Electrical and Computer Engineering  
Iowa State University  
Ames, Iowa, USA  

Shuang Li  
Department of Electrical and Computer Engineering  
Iowa State University  
Ames, Iowa, USA  

---

This package is a collection of MATLAB code used in the paper

**"A Muon-Accelerated Block Coordinate Algorithm for Low-Separation-Rank Tensor Generalized Linear Models"**

The code in this repository is intended to reproduce the numerical results reported in the paper, including synthetic experiments for linear, logistic, and Poisson regression, as well as the VesselMNIST3D experiments.

The code has been tested in MATLAB R2024a.

---

## Repository structure

The MATLAB scripts are named according to the figure numbers in the paper.  
In particular:

- `Fig23_compare_trial_std_v3_time_paper.m`  
  Generates the results for **Figures 2--3** (linear regression: performance across running time).
  
- `Fig4_compare_trial_std_v4_observation.m`  
  Generates the results for **Figure 4** (linear regression: performance vs number of training observations).

  - `Fig56_compare_trial_iteration_logistic_paper.m`  
  Generates the results for **Figures 5--6** (logistic regression: performance across iterations and running time).

- `Fig7_compare_trial_std_observation_logistic_paper.m`  
  Generates the results for **Figure 7** (logistic regression: performance vs number of training observations).

- `Fig89_compare_trial_std_iteration_time_Poisson_paper.m`  
  Generates the results for **Figures 8--9** (Poisson regression: performance across iterations and running time).

- `Fig10_compare_trial_std_observation_Poisson_paper.m`  
  Generates the results for **Figure 10** (Poisson regression: performance vs number of observations).

- `Fig12_Unbalanced_VesselMINST_earlystop_paper.m`  
  Generates the results for **Figure 12** and the corresponding unbalanced VesselMNIST3D experiment.

- `Fig13_Balanced_data_plot_paper.m`  
  Generates the results for **Figure 13** and the corresponding balanced VesselMNIST3D experiment.

---

## Data

### Synthetic data
The synthetic datasets used in the linear, logistic, and Poisson regression experiments are generated within the MATLAB scripts.

### VesselMNIST3D
The VesselMNIST3D dataset is **not stored directly in this repository** because of GitHub file size limits.  
Please download the dataset from the official public source and preprocess it before running the corresponding MATLAB scripts.

Suggested source:
- MedMNIST / VesselMNIST3D
- Zenodo release of VesselMNIST3D

For our experiments, the 3D volumes are used in tensor logistic regression, and for implementation convenience they are also stored in vectorized form.

---

## Reproducibility

To reproduce the figures in the paper:

1. Download or generate the required dataset.
2. Run the MATLAB script corresponding to the target figure number.
3. Adjust file paths or data-loading sections if necessary.

The scripts are organized so that the figure number in the filename matches the figure reported in the paper.

---

## Citation

If you use this code in academic work, please cite our paper.

---

## Contact

If you have any questions or find any bugs, please feel free to contact:

- Xiao Liang: liangx@iastate.edu

