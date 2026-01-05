import numpy as np
import pandas as pd
import pickle

from raking.run_raking import run_raking

# Read dataset
df_obs = pd.read_csv('observations_25.csv')
df_margins = pd.read_csv('margins_25.csv')

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
        margin_names=["_all", 0, 0],
        cov_mat=False,
    )
    df_obs_loc['samples'] = n
    df_raked.append(df_obs_loc)
df_raked = pd.concat(df_raked)

# Save output
with open('results_25_MC.pkl', 'wb') as fp:
    pickle.dump(df_raked, fp)

