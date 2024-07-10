import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('synthetics.pkl', 'rb') as output_file:
    [mu_ij, df_raked, df_x, df_y, covariance_mean, mean_draws, covariance_draws] = pickle.load(output_file)

var1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
var2 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

initial = pd.DataFrame({'var1': var1, \
                         'var2': var2, \
                         'variance': np.arange(0.01, 0.16, 0.01)})

variance = pd.DataFrame({'var1': var1, \
                         'var2': var2, \
                         'variance': np.diag(covariance_mean)})

df_obs = df_raked.drop(columns=['raked_values']).rename(columns={'observations': 'Value'})
df_obs['Type'] = 'Initial'
df_obs['width'] = 1

df_obs = df_obs.merge(initial, how='inner', \
    left_on=['var1', 'var2'], \
    right_on=['var1', 'var2'])

df_raked = df_raked.drop(columns=['observations']).rename(columns={'raked_values': 'Value'})
df_raked['Type'] = 'Raked'
df_raked['width'] = 2

df_raked = df_raked.merge(variance, how='inner', \
    left_on=['var1', 'var2'], \
    right_on=['var1', 'var2'])

df_raked = pd.concat([df_obs, df_raked])

df_raked['Upper'] = df_raked['Value'] + np.sqrt(df_raked['variance'])
df_raked['Lower'] = df_raked['Value'] - np.sqrt(df_raked['variance'])

bar = alt.Chart(df_raked).mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Raked value')),
    alt.X2('Lower:Q'),
    alt.Y('var1:N', axis=alt.Axis(title='Variable 1')),
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart(df_raked).mark_point(
    filled=True
).encode(
    alt.X('Value:Q'),
    alt.Y('var1:N'),
    color=alt.Color('Type:N'),
    shape=alt.Shape('Type:N')
)
chart = alt.layer(point, bar).resolve_scale(
    shape='independent',
    color='independent'
).facet(
    column=alt.Column('var2:N', header=alt.Header(title='Variable 2', titleFontSize=24, labelFontSize=24)),
).configure_axis(
    labelFontSize=24,
    titleFontSize=24
).configure_legend(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('synthetics_raked_values_1.pdf')

bar = alt.Chart(df_raked).mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Raked value')),
    alt.X2('Lower:Q'),
    alt.Y('var2:N', axis=alt.Axis(title='Variable 2')),
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart(df_raked).mark_point(
    filled=True
).encode(
    alt.X('Value:Q'),
    alt.Y('var2:N'),
    color=alt.Color('Type:N'),
    shape=alt.Shape('Type:N')
)
chart = alt.layer(point, bar).resolve_scale(
    shape='independent',
    color='independent'
).facet(
    column=alt.Column('var1:N', header=alt.Header(title='Variable 1', titleFontSize=24, labelFontSize=24)),
).configure_axis(
    labelFontSize=24,
    titleFontSize=24
).configure_legend(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('synthetics_raked_values_2.pdf')

