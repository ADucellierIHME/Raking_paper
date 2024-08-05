# JUQ_paper

This repository contains the python scripts necessary to reproduce the figures for the synthetic dataset in the paper Uncertainty Quantification under Noisy Constraints, with Applications to Raking submitted to JUQ.

The figures for the application requires the raking and uncertainty quantification functions in the scripts raking_methods_1D.py, raking_methods.py and uncertainty_IFT.py, and additional scripts specific to IHME to read the dataset.

The script compare_variance_synthetics.py is used to generate the synthetic dataset and apply the raking and uncertainty quantification methods.

The scripts synthetics_*.py are used to make the figures.

The Jupyter notebook explains how to generate a synthetic dataset and compare the raked values and their uncertainties obtained with the Delta method and Implicit Function Theorem and with the Monte Carlo simulations.

Please install the necessary Python modules using the environment.yml file.
