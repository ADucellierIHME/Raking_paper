# R code to run the Python raking package with reticulate
library(tidyverse)

# Use reticulate and a conda environment to 
library(reticulate)
Sys.setenv("RETICULATE_PYTHON" = '/Users/ducela/anaconda3/envs/env_raking/bin/python')
run_raking <- reticulate::import("raking.run_raking")

# Read dataset
df_obs = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/observations_25.csv')
df_margins = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/margins_25.csv')

# Rake all the draws
df_raked = list()
samples <- df_obs %>% distinct(samples)
for (n in samples$samples) {
  df_obs_loc <- df_obs %>%
    filter(samples==n) %>%
    select(value, cause, race, county)
  df_margins_loc <- df_margins %>%
    filter(samples==n) %>%
    select(cause, value_agg_over_race_county)
  result <- run_raking$run_raking(
    dim='USHD',
    df_obs=df_obs_loc,
    df_margins=list(df_margins_loc),
    var_names=NULL,
    margin_names=list('_all', 0, 0),
    cov_mat=FALSE
  )
  df_raked <- bind_rows(df_raked, result[[1]])
}

