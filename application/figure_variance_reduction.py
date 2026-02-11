import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results.pkl', 'rb') as fp:
    [df, Dphi_y, Dphi_s, sigma] = pickle.load(fp)
df['value'] = df['value'] / df['upper']
df['raked_value'] = df['raked_value'] / df['upper']
df['std'] = df['std'] / df['upper']
df['raked_std'] = np.sqrt(df['std']) / df['upper']

df['cause'] = df['cause'].replace('_all', 'All')
df['cause'] = df['cause'].replace('_comm', 'Comm.')
df['cause'] = df['cause'].replace('_ncd', 'NCD')
df['cause'] = df['cause'].replace('_inj', 'Inj.')
df['race'] = df['race'].replace(1, 'All')
df['race'] = df['race'].replace(2, 'Hisp.')
df['race'] = df['race'].replace(4, 'Black')
df['race'] = df['race'].replace(5, 'White')
df['race'] = df['race'].replace(6, 'AIAN')
df['race'] = df['race'].replace(7, 'API')

min_value = min(df['raked_std'].min(), df['std'].min())
max_value = max(df['raked_std'].max(), df['std'].max())

diagonal = alt.Chart().mark_rule(strokeDash=[8, 8]).encode(
    x=alt.value(0),
    x2=alt.value('width'),
    y=alt.value('height'),
    y2=alt.value(0)
)
mx = alt.Chart().mark_point(size=80, filled=False).encode(
    x=alt.X('std:Q', axis=alt.Axis(title='Standard deviation before raking', format='.1e'), scale=alt.Scale(domain=[min_value, max_value], type='log')),
    y=alt.Y('raked_std:Q', axis=alt.Axis(title='Standard deviation after raking', format='.1e'), scale=alt.Scale(domain=[min_value, max_value], type='log')),
    color=alt.Color('cause:N', legend=alt.Legend(title='Cause')),
    shape=alt.Shape('race:N', legend=alt.Legend(title='Race / ethnicity'))
)
chart = alt.layer(
    diagonal,
    mx,
    data=df
).configure_axis(
    labelFontSize=14,
    titleFontSize=14
).configure_legend(
    labelFontSize=14,
    titleFontSize=14
)
chart.save('variance_reduction.svg')

