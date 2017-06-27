#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import numpy as np
import os
import pandas as pd


from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    assets_path,
    get_filtered_paquid_data,
    )

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    get_historical_mortalite_by_sex,
    get_predicted_mortality_table,
    )


from til_france.targets.population import build_mortality_rates


def get_mortality_from_insee_projection(sexe = None, year = None):
    assert year is not None
    assert sexe in ['female', 'male']
    mortalite_insee = build_mortality_rates()[sexe][year]
    mortalite_insee.name = 'mortalite_insee_{}'.format(year)
    return mortalite_insee


def get_historical_mortality(sexe = None, year = None):
    mortalite_by_sex = get_historical_mortalite_by_sex()
    return (
        mortalite_by_sex[sexe]
        .query('annee == @year')
        .reset_index()
        .loc[:109, ['age', 'mortalite']]
        .astype(dict(age = 'int'))
        .set_index('age')
        .rename(columns = dict(mortalite = 'mortalite_{}'.format(year)))
        )


def plot_paquid_comparison(formula = None, sexe = None):
    assert formula is not None
    mortality_table = get_predicted_mortality_table(formula= formula, sexe = sexe)

    filtered = get_filtered_paquid_data()
    filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)
    filtered['age_group_10'] = 10 * (filtered.age / 10).apply(np.floor).astype('int')
    filtered['age_group_5'] = 5 * ((filtered.age) / 5).apply(np.floor).astype('int')
    if sexe:
        if sexe in ['homme', 'male']:
            sexe_nbr = 1
        elif sexe in ['femme', 'female']:
            sexe_nbr = 2

        filtered = filtered.query('sexe == {}'.format(sexe_nbr))

    test_df = filtered.query('(annee == 2003) & (initial_state != 5)').dropna()
    test_df.initial_state.value_counts()

    test_df.groupby(['age_group_5'])[['final_state']].apply(
        lambda x: 1 - np.sqrt(1 - 1.0 * (x.final_state == 5).sum() / x.count())
        )

    mortalite_insee_2007 = get_mortality_from_insee_projection(sexe = sexe, year = 2007)
    mortalite_1988 = get_historical_mortality(sexe = sexe, year = 1998)

    profile = filtered.query('initial_state != 5').dropna()
    profile.age.round().value_counts()
    profile['age'] = profile.age.round()
        .merge(mortality_table, on =['age', 'initial_state'], how = 'left')
        .groupby(['age'])['mortality']
        .mean()
        )

    plot_data = pd.concat([sample_mortality_profile, mortalite_insee_2007, mortalite_1988], axis = 1)
    plot_data.index.name = 'age'
    ax = plot_data.query('age > 60').plot()
    ax.get_figure().savefig('mortalite.png')

    plot_data['ratio'] = plot_data.eval('mortality / mortalite_insee_2007')

    ax2 = plot_data.query('age > 60').plot(y = 'ratio')
    ax2.get_figure().savefig('ratio_mortalite.png')

    #
    sexe_nbr = 0 if sexe == 'male' else 1
    df = (pd.read_csv(os.path.join(assets_path, 'dependance_niveau.csv'), index_col = 0)
        .query('(period == 2010) and (age >= 60)'))
    assert (df.query('dependance_niveau == -1')['total'] == 0).all()
    mortalite_after_imputation = (df
        .query('dependance_niveau != -1')
        .query('sexe == @sexe_nbr')
        .query('dependance_niveau != 5')
        .rename(columns = dict(dependance_niveau = 'initial_state'))
        .merge(mortality_table, how = 'inner')
        .groupby(['age'])[['total', 'mortality']].apply(lambda x: (x.total * x.mortality).sum() / x.total.sum())
        )
    mortalite_after_imputation.name = 'mortalite_after_imputation'

    plot_data = pd.concat(
        [sample_mortality_profile, mortalite_insee_2007, mortalite_1988, mortalite_after_imputation],
        axis = 1,
        )
    plot_data.index.name = 'age'
    ax = plot_data.query('age > 60').plot()
    ax.get_figure().savefig('mortalite.png')


if __name__ == '__main__':
    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'
    plot_paquid_comparison(formula = formula, sexe = 'female')
