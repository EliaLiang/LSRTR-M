# LSRTR-M

This repository contains the MATLAB code for the paper:

**A Muon-Accelerated Block Coordinate Algorithm for Low-Separation-Rank Tensor Generalized Linear Models**

## Overview

This project implements **LSRTR-M**, a MUON-accelerated block coordinate descent method for low-separation-rank tensor generalized linear models (LSR-TGLMs), and compares it with the baseline **LSRTR** on:

- synthetic linear regression
- synthetic logistic regression
- synthetic Poisson regression
- VesselMNIST3D tensor logistic regression

## Repository Structure

The main scripts in this repository are organized by experiment type.  
Their purposes are consistent with their file names.

### Synthetic experiments
- `compare_trial_std_v3_time_paper.m`  
  Comparison across running time for linear regression experiments.

- `compare_trial_std_v4_observation.m`  
  Comparison across different numbers of training observations for linear regression.

- `compare_trial_iteration_logistic_paper.m`  
  Iteration-based comparison for logistic regression.

- `compare_trial_std_iteration_time_Poisson_paper.m`  
  Iteration/time comparison for Poisson regression.

- `compare_trial_std_observation_Poisson_paper.m`  
  Observation-size comparison for Poisson regression.

- `compare_trial_std_observation_logistic_paper.m`  
  Observation-size comparison for logistic regression.

### VesselMNIST3D experiments
- `Balanced_data_plot_paper.m`  
  Plotting and evaluation for the balanced VesselMNIST3D setting.

### Other files
- `README.md`  
  Project description and reproduction instructions.

## Data

### Synthetic data
Synthetic data are generated within the MATLAB scripts.

### VesselMNIST3D
The VesselMNIST3D dataset is not stored directly in this repository because of GitHub file size limits. Please download it from the official MedMNIST / Zenodo source and preprocess it as described in the paper or in the corresponding scripts.

Suggested source:
- MedMNIST / VesselMNIST3D official website
- Zenodo release for VesselMNIST3D

If needed, you may adapt the preprocessing script used to convert the dataset into MATLAB `.mat` format.


## Reproducibility

To reproduce the results:
1. Run the corresponding MATLAB script for the desired experiment.
2. For VesselMNIST3D experiments, first download and preprocess the dataset.
3. Adjust paths if necessary.


## Citation

If you use this code, please cite the corresponding paper once it becomes available.
