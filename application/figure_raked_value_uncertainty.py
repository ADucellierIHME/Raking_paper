import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results.pkl', 'rb') as fp:
    [df_obs, Dphi_y, Dphi_s, sigma] = pickle.load(fp)

df_initial = df_obs[['cause', 'race', 'county', 'value', 'upper', 'std']]. \
    rename(columns={'value': 'Value', 'upper': 'pop'})
df_initial['Type'] = 'Initial'
df_initial['width'] = 2

df_raked = df_obs[['cause', 'race', 'county', 'raked_value', 'upper', 'variance']]. \
    rename(columns={'raked_value': 'Value', 'upper': 'pop', 'variance': 'std'})
df_raked['std'] = np.sqrt(df_raked['std'])
df_raked['Type'] = 'Raked'
df_raked['width'] = 1

df = pd.concat([df_initial, df_raked])

df['Value'] = df['Value'] / df['pop']
df['Upper'] = df['Value'] + df['std'] / df['pop']
df['Lower'] = df['Value'] - df['std'] / df['pop']

max_x = max(df['Value'])

# Replace names
df['cause'] = df['cause'].replace('_all', 'All')
df['cause'] = df['cause'].replace('_comm', 'Comm.')
df['cause'] = df['cause'].replace('_ncd', 'NCD')
df['cause'] = df['cause'].replace('_inj', 'Inj.')
df['race'] = df['race'].replace(0, 'All')
df['race'] = df['race'].replace(1, 'White')
df['race'] = df['race'].replace(2, 'Black')
df['race'] = df['race'].replace(3, 'AIAN')
df['race'] = df['race'].replace(4, 'API')
df['race'] = df['race'].replace(7, 'Hisp.')
df['county'] = df['county'].replace(301, 'Kent')
df['county'] = df['county'].replace(302, 'New Castle')
df['county'] = df['county'].replace(303, 'Sussex')

# Fix race and county, show values by cause
bar = alt.Chart(df).mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q',
        scale=alt.Scale(domain=[-0.1 * max_x, 1.1 * max_x], zero=False),
        axis=alt.Axis(title='Raked value')),
    alt.X2('Lower:Q'),
    alt.Y('cause:N',
        axis=alt.Axis(title='Cause')),    
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart(df).mark_point(
    filled=True
).encode(
    alt.X('Value:Q'),
    alt.Y('cause:N'),
    color=alt.Color('Type:N'),
    shape=alt.Shape('Type:N')
)
chart = alt.layer(point, bar).resolve_scale(
    shape='independent',
    color='independent'
).facet(
    column=alt.Column('county:N',
        header=alt.Header(title='County', titleFontSize=24, labelFontSize=24)),
    row =alt.Row('race:N',
        header=alt.Header(title='Race', titleFontSize=24, labelFontSize=24))
).configure_axis(
    labelFontSize=24,
    titleFontSize=24
).configure_legend(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('raked_values_with_uncertainty_by_cause.svg')

# Fix cause and county, show values by race
bar = alt.Chart(df).mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q',
        scale=alt.Scale(domain=[-0.1 * max_x, 1.1 * max_x], zero=False),
        axis=alt.Axis(title='Raked value')),
    alt.X2('Lower:Q'),
    alt.Y('race:N',
        axis=alt.Axis(title='Race')),  
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart(df).mark_point(
    filled=True
).encode(
    alt.X('Value:Q'),
    alt.Y('race:N'),
    color=alt.Color('Type:N'),
    shape=alt.Shape('Type:N')
)
chart = alt.layer(point, bar).resolve_scale(
    shape='independent',
    color='independent'
).facet(
    column=alt.Column('county:N',
        header=alt.Header(title='County', titleFontSize=24, labelFontSize=24)),
    row =alt.Row('cause:N',
        header=alt.Header(title='Cause', titleFontSize=24, labelFontSize=24))
).configure_axis(
    labelFontSize=24,
    titleFontSize=24
).configure_legend(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('raked_values_with_uncertainty_by_race.svg')

# Fix cause and race, show values by county
bar = alt.Chart(df).mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q',
        scale=alt.Scale(domain=[-0.1 * max_x, 1.1 * max_x], zero=False),
        axis=alt.Axis(title='Raked value')),
    alt.X2('Lower:Q'),
    alt.Y('county:N',
        axis=alt.Axis(title='County')),
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart(df).mark_point(
    filled=True
).encode(
    alt.X('Value:Q'),
    alt.Y('county:N'),
    color=alt.Color('Type:N'),
    shape=alt.Shape('Type:N')
)
chart = alt.layer(point, bar).resolve_scale(
    shape='independent',
    color='independent'
).facet(
    column=alt.Column('cause:N',
        header=alt.Header(title='Cause', titleFontSize=24, labelFontSize=24)),
    row =alt.Row('race:N',
        header=alt.Header(title='Race', titleFontSize=24, labelFontSize=24))
).configure_axis(
    labelFontSize=24,
    titleFontSize=24
).configure_legend(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('raked_values_with_uncertainty_by_county.svg')

