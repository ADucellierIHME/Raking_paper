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

- Now clone this repository:
```
git clone "https://github.com/ADucellierIHME/Raking_paper.git"
```

You can now use the Python scripts in this repository to reproduce the figures.

The script methods/distances_figure.py is used to make the figure in the Appendix B.1 Impact of Raking Loss of the paper.

The notebook methods/simulation_choice_weights.ipynb is used to make the figure in the Appendix B.2 Impact of Differential Weights of the paper.

The script methods/aggregate_data_figure.py is used to make the figure in the Appendix B.3 Using Aggregates as Observations of the paper.

The script methods/missing_data_figure.py is used to make the figure in the Appendix B.4 Strategies for Missing Data of the paper.

The notebook synthetics/synthetic_example.ipynb is used to make the figures in the Appendic B.5 Uncertainty Quantification of the paper.

The Python scripts in the application directory are used to run the raking on the data and make the figures in the Section 3.3 Application to Mortality Estimates and the Appendices C.2 Additional Results and C.3 Influence of Observations on Raked Values of the paper.

