import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results_25.pkl', 'rb') as fp:
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
bar = alt.Chart().mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q', axis=alt.Axis(title='Mortality rate', format='.1e')),
    alt.X2('Lower:Q'),
    alt.Y('cause:N', axis=alt.Axis(title=None)),    
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart().mark_point(
    filled=False
).encode(
    alt.X('Value:Q', axis=alt.Axis(title='Mortality rate', format='.1e')), 
    alt.Y('cause:N', axis=alt.Axis(title=None)), 
    color=alt.Color('Type:N', legend=alt.Legend(title=None)),
    shape=alt.Shape('Type:N', legend=alt.Legend(title=None))
)
chart = alt.layer(
    bar,
    point,
    data=df   
).facet(
    column=alt.Column('county:N',
        header=alt.Header(title=None, titleFontSize=18, labelFontSize=18)),
    row =alt.Row('race:N',
        sort=['All', 'White', 'Hisp.', 'Black', 'API', 'AIAN'],
        header=alt.Header(title=None, titleFontSize=18, labelFontSize=18))
).configure_axis(
    labelFontSize=18,
    titleFontSize=18
).configure_legend(
    labelFontSize=18,
    titleFontSize=18
)
chart.save('raked_values_with_uncertainty_by_cause_25.svg')

# Fix cause and county, show values by race
bar = alt.Chart().mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q', axis=alt.Axis(title='Mortality rate', format='.1e')),
    alt.X2('Lower:Q'),
    alt.Y('race:N',
          axis=alt.Axis(title=None),
          sort=['All', 'White', 'Hisp.', 'Black', 'API', 'AIAN']
    ),    
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart().mark_point(
    filled=False
).encode(
    alt.X('Value:Q', axis=alt.Axis(title='Mortality rate', format='.1e')), 
    alt.Y('race:N',
          axis=alt.Axis(title=None),
          sort=['All', 'White', 'Hisp.', 'Black', 'API', 'AIAN']
    ), 
    color=alt.Color('Type:N', legend=alt.Legend(title=None)),
    shape=alt.Shape('Type:N', legend=alt.Legend(title=None))
)
chart = alt.layer(
    bar,
    point,
    data=df   
).facet(
    column=alt.Column('county:N',
        header=alt.Header(title=None, titleFontSize=18, labelFontSize=18)),
    row =alt.Row('cause:N',
        sort=['All', 'Comm.', 'NCD', 'Inj.'],
        header=alt.Header(title=None, titleFontSize=18, labelFontSize=18))
).configure_axis(
    labelFontSize=18,
    titleFontSize=18
).configure_legend(
    labelFontSize=18,
    titleFontSize=18
)
chart.save('raked_values_with_uncertainty_by_race_25.svg')

# Fix cause and race, show values by county
bar = alt.Chart().mark_errorbar(clip=True, opacity=0.5).encode(
    alt.X('Upper:Q', axis=alt.Axis(title='Mortality rate', format='.1e')),
    alt.X2('Lower:Q'),
    alt.Y('county:N', axis=alt.Axis(title=None)),    
    color=alt.Color('Type:N', legend=None),
    strokeWidth=alt.StrokeWidth('width:Q', legend=None)
)
point = alt.Chart().mark_point(
    filled=False
).encode(
    alt.X('Value:Q', axis=alt.Axis(title='Mortality rate', format='.1e')), 
    alt.Y('county:N', axis=alt.Axis(title=None)), 
    color=alt.Color('Type:N', legend=alt.Legend(title=None)),
    shape=alt.Shape('Type:N', legend=alt.Legend(title=None))
)
chart = alt.layer(
    bar,
    point,
    data=df   
).facet(
    column=alt.Column('cause:N',
        sort=['All', 'Comm.', 'NCD', 'Inj.'],
        header=alt.Header(title=None, titleFontSize=18, labelFontSize=18)),
    row =alt.Row('race:N',
        sort=['All', 'White', 'Hisp.', 'Black', 'API', 'AIAN'],
        header=alt.Header(title=None, titleFontSize=18, labelFontSize=18))
).configure_axis(
    labelFontSize=18,
    titleFontSize=18
).configure_legend(
    labelFontSize=18,
    titleFontSize=18
)
chart.save('raked_values_with_uncertainty_by_county_25.svg')

