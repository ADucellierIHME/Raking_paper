import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results.pkl', 'rb') as fp:
    [df1, Dphi_y, Dphi_s, sigma] = pickle.load(fp)
df1['workflow'] = '1'

with open('results_4steps.pkl', 'rb') as fp:
    df2 = pickle.load(fp)
df2['workflow'] = '2(K+1)'

df = pd.concat([df1, df2])

df['value'] = df['value'] / df['upper']
df['raked_value'] = df['raked_value'] / df['upper']

min_value = min(df['raked_value'].min(), df['value'].min())
max_value = max(df['raked_value'].max(), df['value'].max())

diagonal = alt.Chart().mark_rule(strokeDash=[8, 8]).encode(
    x=alt.value(0),
    x2=alt.value('width'),
    y=alt.value('height'),
    y2=alt.value(0)
)
mx = alt.Chart().mark_point(size=80, filled=False).encode(
        x=alt.X('value:Q', axis=alt.Axis(title='y', format='.1e'), scale=alt.Scale(domain=[min_value, max_value])),
        y=alt.Y('raked_value:Q', axis=alt.Axis(title='β', format='.1e'), scale=alt.Scale(domain=[min_value, max_value])),
        color=alt.Color('workflow:N', legend=alt.Legend(title='Nb. of pb.')),
        shape=alt.Shape('workflow:N', legend=alt.Legend(title='Nb. of pb.'))
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
chart.save('comparison_workflows_scatter.svg')

