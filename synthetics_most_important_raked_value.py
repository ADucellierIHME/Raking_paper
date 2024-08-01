import altair as alt
import numpy as np
import pandas as pd

import pickle

pd.options.mode.chained_assignment = None

with open('synthetics.pkl', 'rb') as output_file:
    [mu_ij, df_raked, df_x, df_y, covariance_mean, mean_draws, covariance_draws] = pickle.load(output_file)

var1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
var2 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

index = np.argmax(np.abs(df_x.grad_x))
index_raked_1 = df_x.iloc[index].raked_1
index_raked_2 = df_x.iloc[index].raked_2
df_x_loc = df_x.loc[(df_x.raked_1==index_raked_1)&(df_x.raked_2==index_raked_2)]
max_scale = max(abs(df_x_loc['grad_x'].min()), abs(df_x_loc['grad_x'].max()))

base = alt.Chart(df_x_loc).encode(
    x=alt.X('X1:N', axis=alt.Axis(title='X1')),
    y=alt.Y('X2:N', axis=alt.Axis(title='X2')),
)

heatmap = base.mark_rect().encode(
    color=alt.Color('grad_x:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale]),
        legend=alt.Legend(title=['Effect of', 'all obs.']))
)

text = base.mark_text(baseline='middle').encode(
    alt.Text('grad_x:Q', format='.2f')
)

chart = alt.layer(heatmap, text
).properties(
    title='X1 = ' + str(int(index_raked_1)) + ' - X2 = ' + str(int(index_raked_2)),
    width=120,
    height=180
).configure_title(
    fontSize=12
).configure_axis(
    labelFontSize=12,
    titleFontSize=12
).configure_legend(
    labelFontSize=10,
    titleFontSize=10
)
chart.save('synthetics_most_important_raked_value.svg')

