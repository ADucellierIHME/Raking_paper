import altair as alt
import numpy as np
import pandas as pd

import pickle

pd.options.mode.chained_assignment = None

with open('synthetics.pkl', 'rb') as output_file:
    [mu_ij, df_raked, df_x, df_y, covariance_mean, mean_draws, covariance_draws] = pickle.load(output_file)

var1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
var2 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

delta_method = pd.DataFrame({'var1': var1, \
                             'var2': var2, \
                             'delta_method': np.diag(covariance_mean)})

all_draws = pd.DataFrame({'var1': var1, \
                          'var2': var2, \
                          'all_draws': np.diag(covariance_draws)})

df_both = delta_method.merge(all_draws, how='inner', \
    left_on=['var1', 'var2'], \
    right_on=['var1', 'var2'])

min_x = min(df_both['delta_method'].min(), df_both['all_draws'].min())
max_x = max(df_both['delta_method'].max(), df_both['all_draws'].max())

chart = alt.Chart(df_both).mark_point(size=60).encode(
    x=alt.X('all_draws:Q', scale=alt.Scale(domain=[min_x, max_x], zero=False), axis=alt.Axis(title='Using all draws')),
    y=alt.Y('delta_method:Q', scale=alt.Scale(domain=[min_x, max_x], zero=False), axis=alt.Axis(title='Using delta method')),
    color=alt.Color('var1:N', legend=alt.Legend(title='Variable 1')),
    shape=alt.Shape('var2:N', legend=alt.Legend(title='Variable 2'))
).configure_axis(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('synthetics_comparison_variances.pdf')

