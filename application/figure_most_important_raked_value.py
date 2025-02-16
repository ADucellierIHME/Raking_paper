import altair as alt
import numpy as np
import pandas as pd
import pickle

# Name and number of causes, races and counties
causes = ['All', 'Comm.', 'NCD', 'Inj.']
races = ['All', 'White', 'Black', 'AIAN', 'API', 'Hisp.']
counties = ['Kent', 'New Castle', 'Sussex'] 
I = 3
J = 5
K = 3
N = (I + 1) * (J + 1) * K
M = 1 + I + 2 * K + 2 * J * K

# Location of raked value
k = 1
j = 2
i = 3 # Injuries
index = (I + 1) * (J + 1) * int(k) + (I + 1) * int(j) + int(i)

# Read gradient with respect to the data
with open('results.pkl', 'rb') as fp:
    [df_obs, Dphi_y, Dphi_s, sigma] = pickle.load(fp)

obs = Dphi_y[index, :]

# Data frame to store and plot the values
cause_column = []
race_column = []
county_column = []
for k_index in range(0, K):
    for j_index in range(0, J + 1):
        for i_index in range(0, I + 1):
            cause_column.append(causes[i_index])
            race_column.append(races[j_index])
            county_column.append(counties[k_index])
df = pd.DataFrame({'cause': cause_column, \
                    'race': race_column, \
                    'county': county_column})
df['gradient'] = obs
max_scale = max(abs(df['gradient'].min()), abs(df['gradient'].max()))

# Plot for chosen county
df_loc = df.loc[df.county==counties[k]].drop(columns=['county'])
base_county = alt.Chart(df_loc).encode(
    x=alt.X('cause:N', axis=alt.Axis(title='Cause of death')),
    y=alt.Y('race:N', axis=alt.Axis(title='Race and ethnicity')),
)
heatmap_county = base_county.mark_rect().encode(
    color=alt.Color('gradient:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale], reverse=True),
        legend=alt.Legend(title=['Effect of', 'all obs.']))
)
text_county = base_county.mark_text(baseline='middle').encode(
    alt.Text('gradient:Q', format='.2f')
)
chart_county = alt.layer(heatmap_county, text_county
).properties(
    title=counties[k] + ' County',
    width=120,
    height=180)

# Plot for chosen cause
df_loc = df.loc[df.cause==causes[i]].drop(columns=['cause'])
base_cause = alt.Chart(df_loc).encode(
    x=alt.X('county:N', axis= alt.Axis(title='County')),
    y=alt.Y('race:N', axis=alt.Axis(title='Race and ethnicity')),
)
heatmap_cause = base_cause.mark_rect().encode(
    color=alt.Color('gradient:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale], reverse=True),
        legend=alt.Legend(title=['Effect of', 'all obs.']))
)
text_cause = base_cause.mark_text(baseline='middle').encode(
    alt.Text('gradient:Q', format='.2f')
)
chart_cause = alt.layer(heatmap_cause, text_cause
).properties(
    title=causes[i],
    width=90,
    height=180
)

# Plot for chosen race
df_loc = df.loc[df.race==races[j]].drop(columns=['race'])
base_race = alt.Chart(df_loc).encode(
    x=alt.X('county:N', axis= alt.Axis(title='County')),
    y=alt.Y('cause:N', axis=alt.Axis(title='Cause of death')),
)
heatmap_race = base_race.mark_rect().encode(
    color=alt.Color('gradient:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale], reverse=True),
        legend=alt.Legend(title=['Effect of', 'all obs.']))
)
text_race = base_race.mark_text(baseline='middle').encode(
    alt.Text('gradient:Q', format='.2f')
)
chart_race = alt.layer(heatmap_race, text_race
).properties(
    title=races[j],
    width=90,
    height=120
)

# Read gradient with respect to the margins
obs = Dphi_s[index, :]

# Data frame to store and plot the values
cause_column = []
for i_index in range(0, I + 1):
    cause_column.append(causes[i_index])
df = pd.DataFrame({'cause': cause_column})
df['gradient'] = obs[0:(I + 1)]

# Plot for margins
chart_margins = alt.Chart(df).mark_bar().encode(
    x=alt.X('cause:N', axis=alt.Axis(title='Cause of death')),
    y=alt.Y('gradient:Q', axis=alt.Axis(title='Effect of margins')),
).properties(
    title='GBD values',
    width=120,
    height=120
)

chart = alt.hconcat(chart_margins, chart_county, chart_cause, chart_race
).configure_title(
    fontSize=12
).configure_axis(
    labelFontSize=12,
    titleFontSize=12
).configure_legend(
    labelFontSize=10,
    titleFontSize=10
).configure_text(
    fontSize=8
)
chart.save('most_important_raked_value.svg')

