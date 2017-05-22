# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import pkg_resources


assets_path = config_files_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'assets',
    )

matrix_path = os.path.join(assets_path, '3c', 'Proba_transition_3C.xlsx')

df = pd.read_excel(matrix_path, index_col = 0, parse_cols = 'A:G').reset_index()

matrix_3c = df.loc[1:6]
matrix_3c.columns = ['initial'] + range(0, 6)
matrix_3c = (matrix_3c
    .set_index('initial')
    .fillna(0)
    .transpose()
    )

assert (matrix_3c.sum() == 1).all()


matrix_paquid = df.loc[11:16]
matrix_paquid.columns = ['initial'] + range(0, 6)
matrix_paquid = (matrix_paquid
    .set_index('initial')
    .fillna(0)
    .transpose()
    )
assert (matrix_paquid.sum() == 1).all()


transition = matrix_3c.as_matrix()

initial_population = np.array([1, 0, 0, 0, 0, 0])
assert initial_population.sum() == 1
assert transition.dot(initial_population).sum() == 1

initial_age = 65

population_1 = pd.DataFrame({initial_age: initial_population}) / 2
population_2 = pd.DataFrame({initial_age: initial_population}) / 2
population = population_1 + population_2
for year in range(initial_age + 1, 120 + 1):
    if year % 2 == 0:
        population_1[year] = transition.dot(population_1[year - 1])
        population_2[year] = population_2[year - 1]
    else:
        population_2[year] = transition.dot(population_2[year - 1])
        population_1[year] = population_1[year - 1]

    population[year] = population_1[year] + population_2[year]
    assert abs(population[year].sum() - 1.0) < 1e-10, 'year = {}: sum = {}'.format(year, population[year].sum())


alive = population.loc[0:4].sum()
alive.sum()
autonomous = population.loc[0].sum()
autonomous.sum()
dependant = population.loc[1:4].sum()
dependant.sum()
dead = population.loc[5].sum()
dead.sum()
