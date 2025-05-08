import altair as alt
import numpy as np
import pandas as pd

from raking.experimental import  DataBuilder , DualSolver, PrimalSolver

# Step 1: Generate original true data (consistent)
rows, cols, heights = 3, 4, 5   # Define dimensions

# Generate true positive observations (3x4x5 = 60)
original_obs = np.random.rand(rows, cols, heights) * 10   # Random positive values

# Compute consistent marginal sums (12 + 15 + 20 = 47)
row_col_sums = np.sum(original_obs, axis=2)   # Summing over heights (3x4)
row_height_sums = np.sum(original_obs, axis=1)   # Summing over columns (3x5)
col_height_sums = np.sum(original_obs, axis=0)   # Summing over rows (4x5)

# Step 2: Create noisy observations by adding perturbation
noise = (np.random.rand(rows, cols, heights) + 10.0)   # Noise range [0,1]
noisy_obs = original_obs * noise   # Ensure values remain positive

# Step 3: Construct DataFrame treating constraints as observations
obs_data = []
for r in range(rows):
    for c in range(cols):
        for h in range(heights):
            obs_data.append([r, c, h, noisy_obs[r, c, h], original_obs[r, c, h], 1.0])   # Noisy observations

obs_data = pd.DataFrame(obs_data, columns=['row_id', 'col_id', 'height_id', 'obs', 'original', 'weights'])

# Add marginal constraints as observations (weight = 1.0)
marginal_constraints = []

# Row-Col sums (summing across heights) -> 12 constraints (3x4)
for r in range(rows):
    for c in range(cols):
        marginal_constraints.append([r, c, -1, row_col_sums[r, c], row_col_sums[r, c], 1.0])

# Row-Height sums (summing across cols) -> 15 constraints (3x5)
for r in range(rows):
    for h in range(heights):
        marginal_constraints.append([r, -1, h, row_height_sums[r, h], row_height_sums[r, h], 1.0])

# Col-Height sums (summing across rows) -> 20 constraints (4x5)
for c in range(cols):
    for h in range(heights):
        marginal_constraints.append([-1, c, h, col_height_sums[c, h], col_height_sums[c, h], 1.0])

marginal_constraints = pd.DataFrame(marginal_constraints, columns=['row_id', 'col_id', 'height_id', 'obs', 'original', 'weights'])

# First case: Marginal constraints have the same weight as observations

# Combine all data ensuring 60 + 47 = 107 rows
df1 = pd.concat([obs_data, marginal_constraints])

# Rake
data_builder = DataBuilder(
    dim_specs={'row_id': -1, 'col_id': -1, 'height_id': -1},
    value='obs',
    weights='weights',
)
data1 = data_builder.build(df1)

solver1 = DualSolver(distance='entropic', data=data1)
soln1 = solver1.solve()
soln1.rename(columns={'soln': 'soln1'}, inplace=True)

# Second case: Marginal constraints have 2 times the weight as observations

marginal_constraints.loc[:, 'weights'] = 2.0
df2 = pd.concat([obs_data, marginal_constraints])

# Rake
data_builder = DataBuilder(
    dim_specs={'row_id': -1, 'col_id': -1, 'height_id': -1},
    value='obs',
    weights='weights',
)
data2 = data_builder.build(df2)

solver2 = DualSolver(distance='entropic', data=data2)
soln2 = solver2.solve()
soln2.rename(columns={'soln': 'soln2'}, inplace=True)

# Third case: Marginal constraints have 10 times the weight as observations

marginal_constraints.loc[:, 'weights'] = 10.0
df3 = pd.concat([obs_data, marginal_constraints])

# Rake
data_builder = DataBuilder(
    dim_specs={'row_id': -1, 'col_id': -1, 'height_id': -1},
    value='obs',
    weights='weights',
)
data3 = data_builder.build(df3)

solver3 = DualSolver(distance='entropic', data=data3)
soln3 = solver3.solve()
soln3.rename(columns={'soln': 'soln3'}, inplace=True)

# Create data frame for observations
df_obs = df1. \
    merge(soln1, how='inner', on=['row_id', 'col_id', 'height_id']). \
    merge(soln2, how='inner', on=['row_id', 'col_id', 'height_id']). \
    merge(soln3, how='inner', on=['row_id', 'col_id', 'height_id'])

# Compute relative errors
df_obs['relative_error_noisy'] = (df_obs['obs'] - df_obs['original']) / df_obs['original']
df_obs['relative_error_1'] = (df_obs['soln1'] - df_obs['original']) / df_obs['original']
df_obs['relative_error_2'] = (df_obs['soln2'] - df_obs['original']) / df_obs['original']
df_obs['relative_error_3'] = (df_obs['soln3'] - df_obs['original']) / df_obs['original']

# Create a DataFrame for plotting
error_obs = pd.DataFrame({
    'Relative Error': np.concatenate([df_obs['relative_error_noisy'].to_numpy(), \
                                      df_obs['relative_error_1'].to_numpy(), \
                                      df_obs['relative_error_2'].to_numpy(), \
                                      df_obs['relative_error_3'].to_numpy()]),
    'Dataset': ['Initial data'] * len(df_obs) + \
               ['Weight = 1'] * len(df_obs) + \
               ['Weight = 2'] * len(df_obs) + \
               ['Weight = 10'] * len(df_obs)})

# Create a boxplot
chart = alt.Chart(error_obs).mark_boxplot(extent='min-max', size=40, median=True).encode(
    alt.X('Relative Error:Q').scale(type='log'),
    alt.Y('Dataset:N', sort=['Initial data', 'Weight = 1', 'Weight = 2', 'Weight = 10'])
).properties(
    width=500,
    height=200
).configure_axis(
    labelFontSize=14,
    titleFontSize=14
)
chart.save('aggregate_boxplot.svg')

