import altair as alt
import numpy as np
import pandas as pd

from raking.run_raking import run_raking

# Define raking problem)
rng = np.random.default_rng(0)
y = rng.uniform(low=2.0, high=4.0, size=(4, 5))
y = y.flatten(order="F")
weights = 1.0 / np.square(y)
lower = np.repeat(0.5, 20)
upper = np.repeat(4.0, 20)

var1 = np.array([str(i) for i in np.tile(np.arange(1, 5), 5)])
var2 = np.array([str(i) for i in np.repeat(np.arange(1, 6), 4)])
df_obs = pd.DataFrame({'var1': var1, \
                       'var2': var2, \
                       'value': y, \
                       'weights': weights, \
                       'lower': lower, \
                       'upper': upper})

var2 = np.array([str(i) for i in np.arange(1, 6)])
value_agg_over_var1 = np.repeat(4, 5)
df_margins_1 = pd.DataFrame({'var2': var2, 'value_agg_over_var1': value_agg_over_var1})

var1 = np.array([str(i) for i in np.arange(1, 5)])
value_agg_over_var2 = np.repeat(5, 4)
df_margins_2 = pd.DataFrame({'var1': var1, 'value_agg_over_var2': value_agg_over_var2})

# Run chi2 raking
(df_raked, Dphi_y, Dphi_s, sigma) = run_raking(
    dim=2,
    df_obs=df_obs,
    df_margins=[df_margins_1, df_margins_2],
    var_names=['var1', 'var2'],
    method='chi2',
    weights='weights',
    cov_mat=False
)
sum_over_var1 = (
    df_raked.groupby(['var2'])
    .agg({'raked_value': 'sum'})
    .reset_index()
    .merge(df_margins_1, on='var2')
)
assert np.allclose(
    sum_over_var1['raked_value'], sum_over_var1['value_agg_over_var1'], atol=1e-3
), 'The sums over the first variable must match the first margins.'
sum_over_var2 = (
    df_raked.groupby(['var1'])
    .agg({'raked_value': 'sum'})
    .reset_index()
    .merge(df_margins_2, on='var1')
)
assert np.allclose(
    sum_over_var2['raked_value'], sum_over_var2['value_agg_over_var2'], atol=1e-3
), 'The sums over the second variable must match the second margins.'

result_chi2 = df_raked.raked_value.to_numpy()

# Run entropic raking
(df_raked, Dphi_y, Dphi_s, sigma) = run_raking(
    dim=2,
    df_obs=df_obs,
    df_margins=[df_margins_1, df_margins_2],
    var_names=['var1', 'var2'],
    method='entropic',
    weights='weights',
    cov_mat=False
)
sum_over_var1 = (
    df_raked.groupby(['var2'])
    .agg({'raked_value': 'sum'})
    .reset_index()
    .merge(df_margins_1, on='var2')
)
assert np.allclose(
    sum_over_var1['raked_value'], sum_over_var1['value_agg_over_var1'], atol=1e-3
), 'The sums over the first variable must match the first margins.'
sum_over_var2 = (
    df_raked.groupby(['var1'])
    .agg({'raked_value': 'sum'})
    .reset_index()
    .merge(df_margins_2, on='var1')
)
assert np.allclose(
    sum_over_var2['raked_value'], sum_over_var2['value_agg_over_var2'], atol=1e-3
), 'The sums over the second variable must match the second margins.'

result_entropic = df_raked.raked_value.to_numpy()

# Run logit raking
(df_raked, Dphi_y, Dphi_s, sigma) = run_raking(
    dim=2,
    df_obs=df_obs,
    df_margins=[df_margins_1, df_margins_2],
    var_names=['var1', 'var2'],
    method='logit',
    weights='weights',
    lower='lower',
    upper='upper',
    cov_mat=False,
    max_iter=10000
)
sum_over_var1 = (
    df_raked.groupby(['var2'])
    .agg({'raked_value': 'sum'})
    .reset_index()
    .merge(df_margins_1, on='var2')
)
assert np.allclose(
    sum_over_var1['raked_value'], sum_over_var1['value_agg_over_var1'], atol=1e-3
), 'The sums over the first variable must match the first margins.'
sum_over_var2 = (
    df_raked.groupby(['var1'])
    .agg({'raked_value': 'sum'})
    .reset_index()
    .merge(df_margins_2, on='var1')
)
assert np.allclose(
    sum_over_var2['raked_value'], sum_over_var2['value_agg_over_var2'], atol=1e-3
), 'The sums over the second variable must match the second margins.'

result_logit = df_raked.raked_value.to_numpy()

# Concatenate results and plot
df = pd.DataFrame({'initial': np.concatenate([y, y, y]), \
                   'raked': np.concatenate([result_chi2, result_entropic, result_logit]), \
                   'method': ['chi2'] * 20 + ['entropic'] * 20 + ['logit'] * 20})

base = alt.Chart(df)
rule1 = base.mark_rule(strokeDash=[2, 2]).encode(
    y=alt.datum(0)
)
rule2 = base.mark_rule(strokeDash=[2, 2]).encode(
    y=alt.datum(0.5)
)
points = base.mark_point(size=100).encode(
    x = alt.X('initial:Q', axis=alt.Axis(title='Initial values')).scale(zero=False),
    y = alt.Y('raked:Q', axis=alt.Axis(title='Raked values')).scale(zero=False),
    color=alt.Color('method:N').title('Distance'),
    shape=alt.Shape('method:N').title('Distance')
)
chart = alt.layer(
    rule1,
    rule2,
    points
).properties(
    title='Effect of distance on raked values'
).configure_axis(
    labelFontSize=18,
    titleFontSize=18
).configure_title(
    fontSize=18
).configure_legend(
    labelFontSize=16,
    titleFontSize=16
)

chart.save('distance_effect.svg')

