# R code to run the Python raking package with reticulate
library(tidyverse)

# Use reticulate and a conda environment to access the raking package
library(reticulate)
Sys.setenv("RETICULATE_PYTHON" = '/Users/ducela/anaconda3/envs/env_raking/bin/python')
run_raking <- reticulate::import("raking.run_raking")

# Read dataset
df_obs = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/observations.csv')
df_margins = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/margins.csv')

counties <- df_obs %>% distinct(county)

# Compute the mean of the observations
df_obs <- df_obs %>% 
  group_by(cause, race, county, upper) %>%
  summarize(value = mean(value)) %>%
  select(cause, race, county, value, upper)

# Compute the mean of the margins
df_margins <- df_margins %>%
  group_by(cause) %>%
  summarize(value_agg_over_race_county = mean(value_agg_over_race_county)) %>%
  select(cause, value_agg_over_race_county)

# First step: Rake all races all causes mortality by county to state value
df_obs_1 <- df_obs %>%
  filter((cause=='_all') & (race==1)) %>%
  select(cause, race, county, value, upper)
df_margins_1 <- df_margins %>%
  filter(cause=='_all') %>%
  select(value_agg_over_race_county) %>%
  rename(value_agg_over_county = value_agg_over_race_county)
result = run_raking$run_raking(
  dim=1L,
  df_obs=df_obs_1,
  df_margins=list(df_margins_1),
  var_names=list('county'),
  cov_mat=FALSE
)
df_obs_1 = result[[1]]

# Second step: For each county, rake by race all causes mortality to all races value
df_obs_2 = list()
for (n in counties$county) {
  df_obs_2_loc <- df_obs %>%
    filter((cause=='_all') & (race!=1) & (county==n)) %>%
    select(cause, race, county, value, upper)
  df_margins_2 <- df_obs_1 %>%
    filter(county==n) %>%
    select(raked_value) %>%
    rename(value_agg_over_race = raked_value)
  result = run_raking$run_raking(
    dim=1L,
    df_obs=df_obs_2_loc,
    df_margins=list(df_margins_2),
    var_names=list('race'),
    cov_mat=FALSE
  )
  result_loc = result[[1]]
  df_obs_2 <- bind_rows(df_obs_2, result_loc)
}

# Third step: Rake all races mortality by cause and county to all causes and state value
df_obs_3 <- df_obs %>%
  filter((cause!='_all') & (race==1)) %>%
  select(cause, race, county, value, upper)
df_margins_3_1 <- df_obs_1 %>%
  select(county, raked_value) %>%
  rename(value_agg_over_cause = raked_value)
df_margins_3_2 = df_margins %>%
  filter(cause!='_all') %>%
  rename(value_agg_over_county = value_agg_over_race_county)
result = run_raking$run_raking(
  dim=2L,
  df_obs=df_obs_3,
  df_margins=list(df_margins_3_1, df_margins_3_2),
  var_names=list('cause', 'county'),
  cov_mat=FALSE
)
df_obs_3 = result[[1]]

# Fourth step: For each county, rake by race by cause mortality to marginal values
df_obs_4 = list()
for (n in counties$county) {
  df_obs_4_loc <- df_obs %>%
    filter((cause!='_all') & (race!=1) & (county==n)) %>%
    select(cause, race, county, value, upper)
  df_margins_4_1 <- df_obs_2 %>%
    filter(county==n) %>%
    select(county, race, raked_value) %>%
    rename(value_agg_over_cause = raked_value)
  df_margins_4_2 <- df_obs_3 %>%
    filter(county==n) %>%
    select(county, cause, raked_value) %>%
    rename(value_agg_over_race = raked_value)
  result = run_raking$run_raking(
    dim=2L,
    df_obs=df_obs_4_loc,
    df_margins=list(df_margins_4_1, df_margins_4_2),
    var_names=list('cause', 'race'),
    cov_mat=FALSE
  )
  result_loc = result[[1]]
  df_obs_4 <- bind_rows(df_obs_4, result_loc)
}

# Gather results
df_raked = bind_rows(df_obs_1, df_obs_2, df_obs_3, df_obs_4)

# Load the results computed with Python
result_python = py_load_object('/Users/ducela/Documents/Raking/perso/Raking_paper/application/results_4steps.pkl', pickle='pickle')

# Compare the results
print(max(abs(df_raked$raked_value - result_python$raked_value)))

