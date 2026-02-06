import altair as alt
import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

with open('results.pkl', 'rb') as fp:
    [df1, Dphi_y, Dphi_s, sigma] = pickle.load(fp)
df1['Workflow'] = '1 step'

with open('results_4steps.pkl', 'rb') as fp:
    df2 = pickle.load(fp)
df2['Workflow'] = '4 steps'

df = pd.concat([df1, df2])

df['value'] = df['value'] / df['upper']
df['raked_value'] = df['raked_value'] / df['upper']
df['Percentage change'] = 100.0 * np.abs((df['raked_value'] - df['value']) / df['value'])

chart = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('Percentage change:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Workflow:N')
)
chart.save('comparison_workflows_boxplot.svg')

