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
index_var1 = df_x.iloc[index].var1
index_var2 = df_x.iloc[index].var2
df_x_loc = df_x.loc[(df_x.var1==index_var1)&(df_x.var2==index_var2)]
max_scale = max(abs(df_x_loc['grad_x'].min()), abs(df_x_loc['grad_x'].max()))

chart = alt.Chart(df_x_loc).mark_rect().encode(
    x=alt.X('raked_1:N', axis=alt.Axis(title='Variable 1')),
    y=alt.Y('raked_2:N', axis=alt.Axis(title='Variable 2')),
    color=alt.Color('grad_x:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale]),
        legend=alt.Legend(title='Effect of observation'))
).properties(
    title='Variable 1 = ' + str(index_var1) + ' - Variable 2 = ' + str(index_var2),
    width=120,
    height=180)
chart.save('synthetics_most_important_initial_value.pdf')

