import numpy as np
import pandas as pd

from raking.experimental import  DataBuilder , DualSolver, PrimalSolver

observations = pd.read_csv('observations.csv')
margins = pd.read_csv('margins.csv')

observations['weights'] = np.where( \
    observations['value'].isna(), 0.0, 1.0)

margins['race'] = 0
margins['county'] = 0
margins = margins.loc[margins.cause!='_all']
margins.rename(columns={'value_agg_over_race_county': 'value'}, inplace=True)
margins['weights'] = np.inf

df = pd.concat([ \
    observations[['value', 'cause', 'race', 'county', 'weights']], \
    margins[['value', 'cause', 'race', 'county', 'weights']]])

data_builder = DataBuilder(
    dim_specs={'cause': '_all', 'race': 0, 'county': 0},
    value='value',
    weights='weights',
)
data = data_builder.build(df)

solver = DualSolver(distance='entropic', data=data)
soln = solver.solve()

