# -*- coding:utf-8 -*-


from __future__ import division


import os

import numpy
import pandas


from til_france.tests.base import til_france_path


# Data from INSEE projections
data_path = os.path.join(til_france_path, 'param/demo')

sheetname_by_gender = dict(zip(
    ['total', 'male', 'female'],
    ['populationTot', 'populationH', 'populationF']
    ))

population_insee_by_gender = dict(
    (
        gender,
        pandas.read_excel(
            os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
            sheetname = sheetname,
            skiprows = 2,
            header = 2
            )[:109].set_index(u'Âge au 1er janvier')
        )
    for gender, sheetname in sheetname_by_gender.iteritems()
    )

migration_insee_by_gender = dict(
    (
        gender,
        pandas.read_csv(os.path.join(data_path, 'hyp_soldemig{}.csv'.format(suffix)))  # 0 à 109 ans et pas 11 comme dans le fichiex xls
        )
    for gender, suffix in dict(female = 'F', male = 'H').iteritems()
    )

with open(os.path.join(data_path, 'hyp_soldemigH.csv'), 'r') as header_file:
    header = header_file.read().splitlines(True)[:2]

for gender, migration in migration_insee_by_gender.iteritems():

    migration.iloc[0, 0] = 0

    migration_extract_total = migration.iloc[1:, 1:].copy().sum()
    migration_extract = numpy.maximum(migration.iloc[1:, 1:].copy(), 0)
    # Resclaing to deal with emigration
    migration_extract = migration_extract * migration_extract_total/ migration_extract.sum()

    population_insee_by_gender
    total_population = population_insee_by_gender[gender]

    total_population.index = migration_extract.index[:-1]
    migration_extract.iloc[:-1, :] = migration_extract.iloc[:-1, :].copy().astype(float).values / total_population.astype(float).values

    migration.iloc[1:, 1:] = migration_extract.values
    migration.age = migration.age.astype(int)
    suffix = 'F' if gender == 'female' else 'H'
    file_path = os.path.join(data_path, 'hyp_soldemig{}_custom.csv'.format(suffix))
    migration.to_csv(file_path, index = False, header = False)
    with open(file_path, 'r') as input_file:
        data = input_file.read().splitlines(True)

    with open(file_path, 'w') as output_file:
        output_file.writelines(header)
        output_file.writelines(data[1:])
