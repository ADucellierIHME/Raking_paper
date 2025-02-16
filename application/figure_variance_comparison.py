import altair as alt
import numpy as np
import pandas as pd

# All draws
df_all = pd.read_csv('results/Delaware_all_draws/mx_by_cause_by_race_by_county_10_2017_3.csv')
df_all = df_all.groupby(['level', 'area', 'year', 'sex', 'race', 'age', 'cause', 'mcnty', 'state', 'pop']).var().reset_index()

# With IFT and delta method
ages = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
dfs = []
for age in ages:
    df = pd.read_csv('results/variance/mx_by_cause_by_race_by_county_10_2017_3_' + \
        str(age) + '_variance_fixed.csv')
    df['age'] = age
    dfs.append(df)
dfs = pd.concat(dfs)

# Merge
df_both = df_all.merge(dfs, how='inner', \
    left_on=['mcnty', 'age', 'cause', 'race'], \
    right_on=['county', 'age', 'cause', 'race'])
df_both['entropic_distance'] = np.sqrt(df_both['entropic_distance']) 
df_both['entropic'] = np.sqrt(df_both['entropic'])

min_x = min(df_both['entropic_distance'].min(), df_both['entropic'].min())
max_x = max(df_both['entropic_distance'].max(), df_both['entropic'].max())

# Plot
points = alt.Chart(df_both).mark_circle(size=60).encode(
    x=alt.X('entropic_distance:Q', \
        axis=alt.Axis(title='Using all draws'), \
        scale=alt.Scale(domain=[min_x, max_x], zero=False)),
    y=alt.Y('entropic:Q', \
        axis=alt.Axis(title='Using delta method'), \
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
chart.save('comparison_variances.svg')

