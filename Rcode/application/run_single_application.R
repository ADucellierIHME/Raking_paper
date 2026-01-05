# R code to run the Python raking package with reticulate
library(tidyverse)

# Use reticulate and a conda environment to access the raking package
library(reticulate)
Sys.setenv("RETICULATE_PYTHON" = '/Users/ducela/anaconda3/envs/env_raking/bin/python')
run_raking <- reticulate::import("raking.run_raking")

# Read dataset
df_obs = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/observations_25.csv')
df_margins = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/margins_25.csv')

# Compute the standard deviations of the observations
df_std <- df_obs %>% 
  group_by(cause, race, county) %>%
  summarize(std = sqrt(var(value)))
df_obs <- left_join(df_obs, df_std, by=c('cause', 'race', 'county'))

# Rake
result <- run_raking$run_raking(
  dim='USHD',
  df_obs=df_obs,
  df_margins=list(df_margins),
  var_names=NULL,
  margin_names=list('_all', 0, 0),
  cov_mat=TRUE,
  draws='samples'
)

# Get output of the raking
df_obs = result[[1]]
Dphi_y = result[[2]]
Dphi_s = result[[3]]
sigma = result[[4]]

# Load the results computed with Python
result_python = py_load_object('/Users/ducela/Documents/Raking/perso/Raking_paper/application/results_25.pkl', pickle='pickle')

# Compare the results
print(max(abs(df_obs$raked_value - result_python[[1]]$raked_value)))
print(max(abs(df_obs$variance - result_python[[1]]$variance)))
print(max(abs(Dphi_y - result_python[[2]])))
print(max(abs(Dphi_s - result_python[[3]])))
print(max(abs(sigma - result_python[[4]])))
