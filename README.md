# JUQ_paper

This repository contains a Jupyter notebook necessary to reproduce the figures for the synthetic dataset in the paper Uncertainty Quantification under Noisy Constraints, with Applications to Raking submitted to JUQ.

The Jupyter notebook explains how to generate a synthetic dataset and compare the raked values and their uncertainties obtained with the Delta method and Implicit Function Theorem and with the Monte Carlo simulations.

You should first install the raking package through the GitHub repository:

- Clone the raking repository:
```
git clone "https://github.com/ihmeuw-msca/raking"
```

- Go to the new raking directory:
```
cd raking
```

- Create a new pip environment:
```
python3 -m venv .venv
```

- Activate the new environment:
```
source .venv/bin/activate
```

- Upgrade pip:
```
python3 -m pip install --upgrade pip
```

- Install necessary Python packages. I usually install all of these:
```
python3 -m pip install altair dash dash-bootstrap-components ipykernel jupyterlab matplotlib notebook numpy pandas plotly scipy vegafusion vegafusion-python-embed vl-convert-python dash-vega-components pyreadr
```

- Install the raking package:
```
pip install -e .
```

- Create an environment to run the notebook:
```
python3 -m ipykernel install --user --name env_raking --display-name "env_raking"
```

- Launch Jupyterlab:
```
jupyter lab
```

You can then open and run the notebook by changing the kernel to "env_raking" in Jupyterlab.

