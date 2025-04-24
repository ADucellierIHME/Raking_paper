# Raking paper

This repository contains the data and the codes necessary to reproduce the figures in the paper Optimization Perspective on Raking available on [arXiv](https://arxiv.org/abs/2407.20520).  

You should first install the raking package through the GitHub repository:

- Clone the raking repository:
```
git clone "https://github.com/ihmeuw-msca/raking"
```

- Go to the new raking directory:
```
cd raking
```

- Create a new conda environment using the existing yml file:
```
conda env create -f environment.yml 
```

- Activate the new environment:
```
conda activate env_raking
```

- Upgrade pip:
```
pip install --upgrade pip
```

- Install the raking package:
```
pip install -e .
```

Some of the methods described in the paper use an experimental version of the user interface that is not yet available on the [PyPI version](https://pypi.org/project/raking/) of the raking package. To reproduce the corresponding figures, please switch to the experimental branch of the GitHub repository:
```
git checkout experimental
```

The script methods/distances_figure.py is used to make the figure in the part 3.2.1 Raking losses of the paper.

The notebook methods/simulation_choice_weights.ipynb is used to make the figure in the part 3.2.2 Differential weights of the paper.

The script methods/aggregate_data_figure.py is used to make the figure in the part 3.2.3 Aggregate observations of the paper.

The script methods/missing_data_figure.py is used to make the figure in the part 3.2.4 Missing data of the paper.

The notebook synthetics/synthetic_example.ipynb is used to make the figures in the part 3.2.5 2D Raking Example with Uncertainty Quantification of the paper.

The Python scripts in the application directory are used to run the raking on the data and make the figures in part 3.3 Application to mortality estimates of the paper.

