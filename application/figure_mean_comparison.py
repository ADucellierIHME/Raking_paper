import altair as alt
import numpy as np
import pandas as pd

# All draws
df_all = pd.read_csv('results/Delaware_all_draws/mx_by_cause_by_race_by_county_10_2017_3.csv')
df_all = df_all.groupby(['level', 'area', 'year', 'sex', 'race', 'age', 'cause', 'mcnty', 'state']).mean().reset_index()
df_all['all_draws'] = df_all['entropic_distance'] # / df_all['pop']
df_all = df_all.drop(columns=['level', 'area', 'year', 'sex', 'state', \
    'mx', 'sim', 'value', 'pop', \
    'entropic_distance', 'chi2_distance', 'l2_distance', 'logit'])

# With IFT and delta method
df = pd.read_csv('results/Delaware/mx_by_cause_by_race_by_county_10_2017_3.csv')
df['mean'] = df['entropic_distance'] # / df['pop']
df = df.drop(columns=['level', 'area', 'year', 'sex', 'state', \
    'mx', 'sim', 'value', 'pop', \
    'entropic_distance', 'chi2_distance', 'l2_distance', 'logit'])

# Merge
df_both = df_all.merge(df, how='inner', on=['race', 'age', 'cause', 'mcnty'])

min_x = min(df_both['all_draws'].min(), df_both['mean'].min())
max_x = max(df_both['all_draws'].max(), df_both['mean'].max())

# Plot
points = alt.Chart(df_both).mark_circle(size=60).encode(
    x=alt.X('all_draws:Q', \
        axis=alt.Axis(title='Using all draws'), \
        scale=alt.Scale(domain=[min_x, max_x], zero=False)),
    y=alt.Y('mean:Q', \
        axis=alt.Axis(title='Using the mean'), \
        scale=alt.Scale(domain=[min_x, max_x], zero=False))
)
diagonal = alt.Chart().mark_rule(strokeDash=[8, 8]).encode(
    x=alt.value(0),
    x2=alt.value('width'),
    y=alt.value('height'),
    y2=alt.value(0)
)
chart = (diagonal + points).configure_axis(
    labelFontSize=24,
    titleFontSize=24
)
chart.save('comparison_means.svg')

