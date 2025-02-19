import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results_25.pkl', 'rb') as fp:
    [df1, Dphi_y, Dphi_s, sigma] = pickle.load(fp)
df1['workflow'] = '1 step'

with open('results_25_4steps.pkl', 'rb') as fp:
    df2 = pickle.load(fp)
df2['workflow'] = '4 steps'

df = pd.concat([df1, df2])

df['value'] = df['value'] / df['upper']
df['raked_value'] = df['raked_value'] / df['upper']

min_value = min(df['raked_value'].min(), df['value'].min())
max_value = max(df['raked_value'].max(), df['value'].max())

x1, x2 = alt.param(value=min_value), alt.param(value=max_value)
y1, y2 = alt.param(value=min_value), alt.param(value=max_value)
line = alt.Chart().mark_rule(color='grey').encode(
    x=alt.datum(x1, type='quantitative'),
    x2=alt.datum(x2, type='quantitative'),
    y=alt.datum(y1, type='quantitative'),
    y2=alt.datum(y2, type='quantitative')
).add_params(x1, x2, y1, y2)

mx = alt.Chart().mark_point(size=80, filled=True).encode(
        x=alt.X('value:Q', axis=alt.Axis(title='Observations'), scale=alt.Scale(domain=[x1, x2])),
        y=alt.Y('raked_value:Q', axis=alt.Axis(title='Rate of death'), scale=alt.Scale(domain=[y1, y2])),
        color=alt.Color('workflow:N', legend=alt.Legend(title='Workflow')),
        shape=alt.Shape('workflow:N', legend=alt.Legend(title='Workflow'))
    )
chart = alt.layer(line, mx, data=df).configure_axis(
        labelFontSize=18,
        titleFontSize=18
    ).configure_legend(
        labelFontSize=12,
        titleFontSize=12
    )
chart.save('comparison_workflows_scatter_25.svg')

