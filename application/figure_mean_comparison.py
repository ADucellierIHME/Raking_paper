import altair as alt
import numpy as np
import pandas as pd
import pickle

# All draws
with open('results_25_MC.pkl', 'rb') as fp:
    df_all = pickle.load(fp)
df_all = df_all.groupby(['cause', 'race', 'county']).mean().reset_index()
df_all.rename(columns={'raked_value': 'all_draws'}, inplace=True)

# With IFT and delta method
with open('results_25.pkl', 'rb') as fp:
    [df, Dphi_y, Dphi_s, sigma] = pickle.load(fp)
df.rename(columns={'raked_value': 'mean'}, inplace=True)

# Merge
df_both = df_all.merge(df, how='inner', on=['cause', 'race', 'county'])

min_x = min(df_both['all_draws'].min(), df_both['mean'].min())
max_x = max(df_both['all_draws'].max(), df_both['mean'].max())

# Plot
points = alt.Chart(df_both).mark_circle(size=60).encode(
    x=alt.X('all_draws:Q', \
        axis=alt.Axis(title='Σ𝞿(yᵢ,sᵢ) / N'), \
        scale=alt.Scale(domain=[min_x, max_x], zero=False)),
    y=alt.Y('mean:Q', \
        axis=alt.Axis(title='𝞿(y\u0305,s\u0305)'), \
        scale=alt.Scale(domain=[min_x, max_x], zero=False))
)
diagonal = alt.Chart().mark_rule(strokeDash=[8, 8]).encode(
    x=alt.value(0),
    x2=alt.value('width'),
    y=alt.value('height'),
    y2=alt.value(0)
)
chart = (diagonal + points).configure_axis(
    labelFontSize=18,
    titleFontSize=18
)
chart.save('comparison_means_25.svg')

