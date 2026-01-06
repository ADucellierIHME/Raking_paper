import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results_25.pkl', 'rb') as fp:
    [df, Dphi_y, Dphi_s, sigma] = pickle.load(fp)

df['value'] = df['value'] / df['upper']
df['raked_value'] = df['raked_value'] / df['upper']
df['Percentage change (%)'] = 100.0 * np.abs((df['raked_value'] - df['value']) / df['value'])

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

# Fix race and county, boxplot by cause
chart_cause = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('cause:N', axis=alt.Axis(title='Cause'), sort=['All', 'Inj.', 'NCD', 'Comm.']),
    y=alt.Y('Percentage change (%):Q', scale=alt.Scale(zero=False))
).properties(
    width=120,
    height=150
)

# Fix cause and county, boxplot by race
chart_race = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('race:N', axis=alt.Axis(title='Race'), sort=['All', 'White', 'Hisp.', 'Black', 'API', 'AIAN']),
    y=alt.Y('Percentage change (%):Q', axis=alt.Axis(title=None, labels=False), scale=alt.Scale(zero=False))
).properties(
    width=180,
    height=150
)

# Fix cause and race, show values by county
chart_county = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('county:N', axis=alt.Axis(title='County'), sort=['New Castle', 'Kent', 'Sussex']),
    y=alt.Y('Percentage change (%):Q', axis=alt.Axis(title=None, labels=False), scale=alt.Scale(zero=False))
).properties(
    width=90,
    height=150
)

chart = alt.hconcat(
    chart_cause, chart_race, chart_county
)
chart.save('raked_values_boxplot.svg')
