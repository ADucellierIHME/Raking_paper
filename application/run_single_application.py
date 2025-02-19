import numpy as np
import pandas as pd
import pickle

from raking.run_raking import run_raking

# Read dataset
df_obs = pd.read_csv('observations_25.csv')
df_margins = pd.read_csv('margins_25.csv')

# Compute the standard deviations of the observations
df_std = df_obs.groupby(['cause', 'race', 'county']).std().reset_index()
df_std = df_std[['cause', 'race', 'county', 'value']].rename(columns={'value': 'std'})
df_obs = df_obs.merge(df_std, on=['cause', 'race', 'county'], how='left')

# Rake
(df_obs, Dphi_y, Dphi_s, sigma) = run_raking(
    dim='USHD',
    df_obs=df_obs,
    df_margins=[df_margins],
    var_names=None,
    cov_mat=True,
    draws='samples'
)

# Save output
with open('results_25.pkl', 'wb') as fp:
    pickle.dump([df_obs, Dphi_y, Dphi_s, sigma], fp)

