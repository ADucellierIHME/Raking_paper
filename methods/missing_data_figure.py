import altair as alt
import numpy as np
import pandas as pd

from raking.experimental import  DataBuilder , DualSolver, PrimalSolver

observations = pd.read_csv('observations.csv')
margins = pd.read_csv('margins.csv')

observations['weights'] = np.where( \
    observations['value'].isna(), 0.0, 1.0)

margins['race'] = 0
margins['county'] = 0
margins = margins.loc[margins.cause!='_all']
margins.rename(columns={'value_agg_over_race_county': 'value'}, inplace=True)
margins['weights'] = np.inf

df = pd.concat([ \
    observations[['value', 'cause', 'race', 'county', 'weights']], \
    margins[['value', 'cause', 'race', 'county', 'weights']]])

# First raking with missing values

data_builder = DataBuilder(
    dim_specs={'cause': '_all', 'race': 0, 'county': 0},
    value='value',
    weights='weights',
)
data = data_builder.build(df)

solver = DualSolver(distance='entropic', data=data)
soln = solver.solve()

sum_over_cause = soln.groupby(['race', 'county']).agg({'soln': 'sum'}).reset_index()
sum_over_cause['cause'] = '_all'

sum_over_race = soln.groupby(['cause', 'county']).agg({'soln': 'sum'}).reset_index()
sum_over_race['race'] = 0

sum_over_cause_race = soln.groupby(['county']).agg({'soln': 'sum'}).reset_index()
sum_over_cause_race['cause'] = '_all'
sum_over_cause_race['race'] = 0

result = pd.concat([soln, sum_over_cause, sum_over_race, sum_over_cause_race]). \
    merge(df, how='inner', on=['cause', 'race', 'county'])

# Second raking: Replace missing values by 0

df2 = df.copy()
df2.fillna(0.0, inplace=True)
df2['weights'] = df2['weights'].replace(0.0, 1.0)

data2 = data_builder.build(df2)

solver2 = DualSolver(distance='entropic', data=data2)
soln2 = solver2.solve()

sum_over_cause = soln2.groupby(['race', 'county']).agg({'soln': 'sum'}).reset_index()
sum_over_cause['cause'] = '_all'

sum_over_race = soln2.groupby(['cause', 'county']).agg({'soln': 'sum'}).reset_index()
sum_over_race['race'] = 0

sum_over_cause_race = soln2.groupby(['county']).agg({'soln': 'sum'}).reset_index()
sum_over_cause_race['cause'] = '_all'
sum_over_cause_race['race'] = 0

result2 = pd.concat([soln2, sum_over_cause, sum_over_race, sum_over_cause_race]). \
    merge(df, how='inner', on=['cause', 'race', 'county'])

# Plot and compare
result['Missing values'] = 'Missing'
result2['Missing values'] = 'Filled with 0'
df_res = pd.concat([result, result2])

chart = alt.Chart(df_res).mark_circle().encode(
    x = alt.X('value:Q', axis=alt.Axis(title='Initial values')),
    y = alt.Y('soln:Q', axis=alt.Axis(title='Raked values')),
    color=alt.Color('Missing values:N')
).configure_axis(
    labelFontSize=18,
    titleFontSize=18
).configure_title(
    fontSize=18
).configure_legend(
    labelFontSize=16,
    titleFontSize=16
)

chart.save('missing_values.svg')

