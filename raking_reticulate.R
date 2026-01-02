


library(reticulate)
library(tidyverse)
Sys.setenv("RETICULATE_PYTHON" = '/Users/ducela/anaconda3/envs/env_raking/bin/python')
run_raking <- reticulate::import("raking.run_raking")

df_obs = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/observations_25.csv')
df_margins = read_csv('/Users/ducela/Documents/Raking/perso/Raking_paper/application/margins_25.csv')

df_std <- df_obs %>% 
  group_by(cause, race, county) %>%
  summarize(std = sqrt(var(value)))
df_obs <- left_join(df_obs, df_std, by=c('cause', 'race', 'county'))

result <- run_raking$run_raking(
  dim='USHD',
  df_obs=df_obs,
  df_margins=list(df_margins),
  var_names=NULL,
  margin_names=list('_all', 0, 0),
  cov_mat=TRUE,
  draws='samples'
)

