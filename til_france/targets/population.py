# -*- coding:utf-8 -*-


from __future__ import division


import logging
import numpy
import os
import pkg_resources
import pandas


from til_core.config import Config


til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )


log = logging.getLogger(__name__)


def get_data_frame_insee(gender, by = 'age_group'):
    data_path = os.path.join(til_france_path, 'param/demo/projpop0760_FECcentESPcentMIGcent.xls')
    sheetname_by_gender = dict(zip(
        ['total', 'male', 'female'],
        ['populationTot', 'populationH', 'populationF']
        ))
    population_insee = pandas.read_excel(
        data_path, sheetname = sheetname_by_gender[gender], skiprows = 2, header = 2)[:109].set_index(
            u'Âge au 1er janvier')

    population_insee.reset_index(inplace = True)
    population_insee.drop(population_insee.columns[0], axis = 1, inplace = True)
    population_insee.index.name = 'age'
    population_insee.columns = population_insee.columns + (-1)  # Passage en âge en fin d'année
    if by == 'age':
        return population_insee
    population_insee.reset_index(inplace = True)
    population_insee['age_group'] = population_insee.age // 10
    population_insee.drop('age', axis = 1, inplace = True)
    data_frame_insee = population_insee.groupby(['age_group']).sum()
    return data_frame_insee


def build_deaths(to_csv = False, input_dir = None, uniform_weight = 200):
    # Data from INSEE projections
    data_path = os.path.join(til_france_path, 'param', 'demo')

    sheetname_by_gender = dict(zip(
        ['male', 'female'],
        ['nbre_decesH', 'nbre_decesF']
        ))
    deaths_by_gender = dict(
        (
            gender,
            pandas.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheetname = sheetname, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for gender, sheetname in sheetname_by_gender.iteritems()
        )
    for df in deaths_by_gender.values():
        del df[u"Âge atteint dans l'année"]

    if to_csv:
        if input_dir is None:
            config = Config()
            input_dir = config.get('til', 'input_dir')

        for gender in ['male', 'female']:
            gender_letter = 'H' if gender == 'male' else 'F'
            file_path = os.path.join(input_dir, 'parameters', 'population', 'deces{}.csv'.format(gender_letter))
            check_population_directory_existence(input_dir)
            df = deaths_by_gender[gender]
            columns_for_liam = ['age', 'period'] + [''] * (len(df.columns) - 1)
            first_row = ','.join([''] + [str(year) for year in df.columns])
            header = ','.join(columns_for_liam) + '\n' + first_row + '\n'
            df.dropna(inplace = True)
            df.drop(109, inplace = True)
            df = (df / 200).round()
            df.to_csv(file_path, index = True, header = False)
            with open(file_path, 'r') as input_file:
                data = input_file.read().splitlines(True)

            with open(file_path, 'w') as output_file:
                output_file.writelines(header)
                output_file.writelines(data)

    else:
        return deaths_by_gender


def build_mortality_rates(to_csv = False, input_dir = None):
    # Data from INSEE projections
    data_path = os.path.join(til_france_path, 'param', 'demo')

    sheetname_by_gender = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_gender = dict(
        (
            gender,
            pandas.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheetname = sheetname, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for gender, sheetname in sheetname_by_gender.iteritems()
        )

    for df in mortality_by_gender.values():
        del df[u"Âge atteint dans l'année"]

    for gender in ['male', 'female']:
        mortality_by_gender[gender] = mortality_by_gender[gender] / 1e4

    if to_csv:
        if input_dir is None:
            config = Config()
            input_dir = config.get('til', 'input_dir')

        for gender in ['male', 'female']:
            gender_letter = 'H' if gender == 'male' else 'F'
            file_path = os.path.join(input_dir, 'parameters', 'population', 'hyp_mortalite{}.csv'.format(gender_letter))
            check_population_directory_existence(input_dir)
            df = mortality_by_gender[gender]
            columns_for_liam = ['age', 'period'] + [''] * (len(df.columns) - 1)
            first_row = ','.join([''] + [str(year) for year in df.columns])
            header = ','.join(columns_for_liam) + '\n' + first_row + '\n'
            df.to_csv(file_path, index = True, header = False)
            with open(file_path, 'r') as input_file:
                data = input_file.read().splitlines(True)

            with open(file_path, 'w') as output_file:
                output_file.writelines(header)
                output_file.writelines(data)

    else:
        return mortality_by_gender


def rescale_migration(input_dir = None):
    u'''Calcule le solde migratoire net utilisé pour caler l'immigration
    Les valeurs sont stockées dans parameters/input/hyp_soldemig[HF]_custom.csv
    '''
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
            pandas.read_csv(os.path.join(data_path, 'hyp_soldemig{}.csv'.format(suffix)))
            # 0 à 109 ans et pas 11 comme dans le fichiex xls
            )
        for gender, suffix in dict(female = 'F', male = 'H').iteritems()
        )

    with open(os.path.join(data_path, 'hyp_soldemigH.csv'), 'r') as header_file:
        header = header_file.read().splitlines(True)[:2]

    if input_dir is None:
        config = Config()
        input_dir = config.get('til', 'input_dir')

    for gender, migration in migration_insee_by_gender.iteritems():

        migration.iloc[0, 0] = 0

        migration_extract_total = migration.iloc[1:, 1:].copy().sum()
        migration_extract = numpy.maximum(migration.iloc[1:, 1:].copy(), 0)
        # Resclaing to deal with emigration
        migration_extract = migration_extract * migration_extract_total / migration_extract.sum()

        total_population = population_insee_by_gender[gender]

        total_population.index = migration_extract.index[:-1]
        migration_extract.iloc[:-1, :] = (
            migration_extract.iloc[:-1, :].copy().astype(float).values / total_population.astype(float).values
            )

        migration.iloc[1:, 1:] = migration_extract.values
        migration.age = migration.age.astype(int)
        suffix = 'F' if gender == 'female' else 'H'
        check_population_directory_existence(input_dir)
        file_path = os.path.join(input_dir, 'parameters', 'population', 'hyp_soldemig{}_custom.csv'.format(suffix))
        migration.to_csv(file_path, index = False, header = False)
        with open(file_path, 'r') as input_file:
            data = input_file.read().splitlines(True)

        with open(file_path, 'w') as output_file:
            output_file.writelines(header)
            output_file.writelines(data[1:])


def check_population_directory_existence(input_dir):
    population_directory_path = os.path.join(os.path.join(input_dir, 'parameters', 'population'))
    if not os.path.exists(population_directory_path):
        log.info('Creating directory {}'.format(population_directory_path))
        os.makedirs(population_directory_path)
