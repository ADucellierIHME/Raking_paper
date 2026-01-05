import numpy as np
import pandas as pd
import pickle

from raking.run_raking import run_raking

# Read dataset
df_obs = pd.read_csv('observations_25.csv')
df_margins = pd.read_csv('margins_25.csv')

counties = df_obs.county.unique().tolist()

# Compute the mean of the observations
df_obs = df_obs.groupby(['cause', 'race', 'county', 'upper']).mean().reset_index()
df_obs = df_obs[['cause', 'race', 'county', 'value', 'upper']]

# Compute the mean of the margins
df_margins = df_margins.groupby(['cause']).mean().reset_index()
df_margins = df_margins[['cause', 'value_agg_over_race_county']]

# First step: Rake all races all causes mortality by county to state value
df_obs_1 = df_obs.loc[(df_obs.cause=='_all')&(df_obs.race==0)]
df_obs_1 = df_obs_1[['county', 'value', 'upper']]
df_margins_1 = df_margins.loc[df_margins.cause=='_all']
df_margins_1 = df_margins_1[['value_agg_over_race_county']]. \
    rename(columns={'value_agg_over_race_county': 'value_agg_over_county'})
(df_obs_1, dummy1, dummy2, dummy3) = run_raking(
    dim=1,
    df_obs=df_obs_1,
    df_margins=[df_margins_1],
    var_names=['county'],
    cov_mat=False
)

# Second step: For each county, rake by race all causes mortality to all races value
df_obs_2 = []
for county in counties:
    df_obs_2_loc = df_obs.loc[(df_obs.cause=='_all')&(df_obs.race!=0)&(df_obs.county==county)]
    df_obs_2_loc = df_obs_2_loc[['county', 'race', 'value', 'upper']]
    df_margins_2 = df_obs_1.loc[df_obs_1.county==county][['raked_value']]. \
        rename(columns={'raked_value': 'value_agg_over_race'})
    (df_obs_2_loc, dummy1, dummy2, dummy3) = run_raking(
        dim=1,
        df_obs=df_obs_2_loc,
        df_margins=[df_margins_2],
        var_names=['race'],
        cov_mat=False
    )
    df_obs_2.append(df_obs_2_loc)
df_obs_2 = pd.concat(df_obs_2)

# Third step: Rake all races mortality by cause and county to all causes and state value
df_obs_3 = df_obs.loc[(df_obs.cause!='_all')&(df_obs.race==0)]
df_obs_3 = df_obs_3[['cause', 'county', 'value', 'upper']]
df_margins_3_1 = df_obs_1[['county', 'raked_value']]. \
    rename(columns={'raked_value': 'value_agg_over_cause'})
df_margins_3_2 = df_margins.loc[df_margins.cause!='_all']. \
    rename(columns={'value_agg_over_race_county': 'value_agg_over_county'})
(df_obs_3, dummy1, dummy2, dummy3) = run_raking(
    dim=2,
    df_obs=df_obs_3,
    df_margins=[df_margins_3_1, df_margins_3_2],
    var_names=['cause', 'county'],
    cov_mat=False
)

# Fourth step: For each county, rake by race by cause mortality to marginal values
df_obs_4 = []
for county in counties:
    df_obs_4_loc = df_obs.loc[(df_obs.cause!='_all')&(df_obs.race!=0)&(df_obs.county==county)]
    df_obs_4_loc = df_obs_4_loc[['cause', 'race', 'county', 'value', 'upper']]
    df_margins_4_1 = df_obs_2.loc[df_obs_2.county==county][['county', 'race', 'raked_value']]. \
        rename(columns={'raked_value': 'value_agg_over_cause'})
    df_margins_4_2 = df_obs_3.loc[df_obs_3.county==county][['county', 'cause', 'raked_value']]. \
        rename(columns={'raked_value': 'value_agg_over_race'})
    (df_obs_4_loc, dummy1, dummy2, dummy3) = run_raking(
        dim=2,
        df_obs=df_obs_4_loc,
        df_margins=[df_margins_4_1, df_margins_4_2],
        var_names=['cause', 'race'],
        cov_mat=False
    )
    df_obs_4.append(df_obs_4_loc)
df_obs_4 = pd.concat(df_obs_4)

# Gather results and save
df_obs_1['cause'] = '_all'
df_obs_1['race'] = 0
df_obs_2['cause'] = '_all'
df_obs_3['race'] = 0
df_raked = pd.concat([df_obs_1, df_obs_2, df_obs_3, df_obs_4])
with open('results_25_4steps.pkl', 'wb') as fp:
    pickle.dump(df_raked, fp)

