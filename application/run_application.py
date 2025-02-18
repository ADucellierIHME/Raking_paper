import numpy as np
import pandas as pd
import pickle

from raking.run_raking import run_raking

# Read dataset
df_obs = pd.read_csv('observations_0.csv')
df_margins = pd.read_csv('margins_0.csv')

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
with open('results_0.pkl', 'wb') as fp:
    pickle.dump([df_obs, Dphi_y, Dphi_s, sigma], fp)

# Read dataset
df_obs = pd.read_csv('observations_0.csv')
df_margins = pd.read_csv('margins_0.csv')

# Rake all the draws
df_raked = []
for n in df_obs.samples.unique():
    df_obs_loc = df_obs.loc[df_obs['samples']==n][['value', 'cause', 'race', 'county']]
    df_margins_loc = df_margins.loc[df_margins['samples']==n][['cause', 'value_agg_over_race_county']]
    (df_obs_loc, dummy1, dummy2, dummy3) = run_raking(
        dim='USHD',
        df_obs=df_obs_loc,
        df_margins=[df_margins_loc],
        var_names=None,
        cov_mat=False,
    )
    df_raked.append(df_obs_loc)
df_raked = pd.concat(df_raked)

# Save output
with open('results_0_MC.pkl', 'wb') as fp:
    pickle.dump(df_raked, fp)

