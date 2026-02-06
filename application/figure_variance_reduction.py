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
    y=alt.Y('raked_std:Q', axis=alt.Axis(title='Standard deviation after raking', format='.1e'), scale=alt.Scale(domain=[min_value, max_value], type='log'))
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

