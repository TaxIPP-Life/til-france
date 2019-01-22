# -*- coding: utf-8 -*-


import os


import pandas as pd

from til_france.plot.population import get_insee_projection

from til_france.model.options.dependance_RT.life_expectancy.share.paths_prog import (
    til_france_path,
    )


def get_insee_projected_mortality_raw():
    """
        Get mortality data from INSEE projections
    """
    data_path = os.path.join(til_france_path, 'param', 'demo')
    sheet_name_by_sex = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_sex = dict(
        (
            sex,
            pd.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheet_name = sheet_name, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for sex, sheet_name in sheet_name_by_sex.items()
        )

    for df in mortality_by_sex.values():
        del df[u"Âge atteint dans l'année"]
        df.index.name = 'age'

    mortality_insee = None
    for sex in ['male', 'female']:
        mortality_sex = ((mortality_by_sex[sex] / 1e4)
            .reset_index()
            )
        mortality_sex = pd.melt(
            mortality_sex,
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'mortality_insee'
            )
        mortality_sex['sex'] = sex
        mortality_sex.rename(columns = dict(annee = 'year'), inplace = True)
        mortality_insee = pd.concat([mortality_insee, mortality_sex])

    return mortality_insee.set_index(['sex', 'age', 'year'])


def get_insee_projected_mortality():
    mortality_insee_next_period = (get_insee_projected_mortality_raw()
        .sort_values(by = ['sex', 'year', 'age'])
        .groupby(['sex', 'year']).shift(-1)
        .sort_values(by = ['sex', 'age', 'year'])
        .groupby(['sex', 'age']).shift(-1)
        )

    mortality_insee = get_insee_projected_mortality_raw()
    mortality_insee['mortality_insee_next_period'] = mortality_insee_next_period.mortality_insee
    mortality_insee['mortalite_2_year_insee'] = 0
    mortality_insee.loc[
        mortality_insee.mortality_insee_next_period.notnull(),
        'mortalite_2_year_insee'
        ] = 1 - (1 - mortality_insee.mortality_insee) * (1 - mortality_insee.mortality_insee_next_period)

    mortality_insee.loc[
        mortality_insee.mortality_insee_next_period.isnull(),
        'mortalite_2_year_insee'
        ] = 1 - (1 - mortality_insee.mortality_insee) ** 2

    return mortality_insee


def get_insee_projected_population():
    """
        Get population data from INSEE projections
    """
    population_by_sex = dict(
        (
            sex,
            get_insee_projection('population', sex)
            )
        for sex in ['male', 'female']
        )

    for df in population_by_sex.values():
        df.index.name = 'age'

    population = None
    for sex in ['male', 'female']:
        population_sex = ((population_by_sex[sex])
            .reset_index()
            )
        population_sex = pd.melt(
            population_sex,
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'population'
            )
        population_sex['sex'] = sex
        population_sex.rename(columns = dict(annee = 'year'), inplace = True)
        population = pd.concat([population, population_sex])

    # Fix age values
    population.age = population.age.replace({'108 et +': 108}).astype(int)
    return population.set_index(['sex', 'age', 'year'])
