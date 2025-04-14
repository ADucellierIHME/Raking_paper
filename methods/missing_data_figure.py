import altair as alt
import numpy as np
import pandas as pd

from raking.experimental import  DataBuilder , DualSolver, PrimalSolver

pd.options.mode.chained_assignment = None

# Read dataset
observations = pd.read_csv('observations.csv')
margins = pd.read_csv('margins.csv')

# Raking with all data
observations['weights'] = 1.0

margins['race'] = 0
margins['county'] = 0
margins = margins.loc[margins.cause!='_all']
margins.rename(columns={'value_agg_over_race_county': 'value'}, inplace=True)
margins['weights'] = np.inf

df = pd.concat([ \
    observations[['value', 'cause', 'race', 'county', 'weights']], \
    margins[['value', 'cause', 'race', 'county', 'weights']]]).reset_index()

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

sum_over_race_county = soln.groupby(['cause']).agg({'soln': 'sum'}). \
    merge(margins, on=['cause'], how='inner').reset_index()
assert np.allclose(sum_over_race_county.value.to_numpy(), \
                   sum_over_race_county.soln.to_numpy())

result = pd.concat([soln, sum_over_cause, sum_over_race, sum_over_cause_race]). \
    merge(df, how='inner', on=['cause', 'race', 'county'])

# Raking with missing values replaced by 0
df2 = df.copy(deep=True)
df2['value'].loc[(df2.race==3)&(df2.county==533)] = 0.0

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

sum_over_race_county = soln2.groupby(['cause']).agg({'soln': 'sum'}). \
    merge(margins, on=['cause'], how='inner').reset_index()
assert np.allclose(sum_over_race_county.value.to_numpy(), \
                   sum_over_race_county.soln.to_numpy())

result2 = pd.concat([soln2, sum_over_cause, sum_over_race, sum_over_cause_race]). \
    merge(df, how='inner', on=['cause', 'race', 'county'])

# Raking with missing values treated as missing
df3 = df.copy(deep=True)
df3['value'].loc[(df3.race==3)&(df3.county==533)] = np.nan
df3['weights'].loc[(df3.race==3)&(df3.county==533)] = 0.0
df3 = df3.loc[(df3.cause!='_all')|(df3.race!=3)|(df3.county!=533)]
df3.reset_index(drop=True, inplace=True)

data3 = data_builder.build(df3)

solver3 = DualSolver(distance='entropic', data=data3)
soln3 = solver3.solve()

sum_over_cause = soln3.groupby(['race', 'county']).agg({'soln': 'sum'}).reset_index()
sum_over_cause['cause'] = '_all'

sum_over_race = soln3.groupby(['cause', 'county']).agg({'soln': 'sum'}).reset_index()
sum_over_race['race'] = 0

sum_over_cause_race = soln3.groupby(['county']).agg({'soln': 'sum'}).reset_index()
sum_over_cause_race['cause'] = '_all'
sum_over_cause_race['race'] = 0

sum_over_race_county = soln2.groupby(['cause']).agg({'soln': 'sum'}). \
    merge(margins, on=['cause'], how='inner').reset_index()
assert np.allclose(sum_over_race_county.value.to_numpy(), \
                   sum_over_race_county.soln.to_numpy())

result3 = pd.concat([soln3, sum_over_cause, sum_over_race, sum_over_cause_race]). \
    merge(df, how='inner', on=['cause', 'race', 'county'])

# Plot and compare
result['Missing values'] = 'Present'
result2['Missing values'] = 'Filled with 0'
result3['Missing values'] = 'Missing'
df_res = pd.concat([result, result2, result3])

chart1 = alt.Chart(df_res).mark_circle().encode(
    x = alt.X('value:Q', axis=alt.Axis(title='Initial values')),
    y = alt.Y('soln:Q', axis=alt.Axis(title='Raked values')),
    color=alt.Color('Missing values:N')
).properties(
    title='All observations'
)

chart2 = alt.Chart(df_res.loc[(df_res.race==3)&(df_res.county==533)]).mark_circle().encode(
    x = alt.X('value:Q', axis=alt.Axis(title='Initial values')),
    y = alt.Y('soln:Q', axis=alt.Axis(title='Raked values')),
    color=alt.Color('Missing values:N')
).properties(
    title='Missing observations'
)

chart = alt.hconcat(
    chart1,
    chart2
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

