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
    build_mortality_calibrated_targets,
    get_historical_mortality,
    get_predicted_mortality_table,
    get_transitions_from_formula,
    )


from til_france.targets.population import build_mortality_rates


def get_mortality_from_insee_projection(year = None):
    assert year is not None
    mortalite_insee = pd.concat([
        (build_mortality_rates()[sex][year]
            .reset_index()
            .rename(columns = dict(index = 'age'))
            .assign(sex = sex)
            ) for sex in ['male', 'female']
        ])
    mortalite_insee['mortalite_insee_{}'.format(year)] = mortalite_insee[year]
    del mortalite_insee[year]
    return mortalite_insee.set_index(['sex', 'age'])


def extract_historical_mortality(year = None):
    return (get_historical_mortality()
        .query('(annee == @year)')
        .query('age <= 110')[['sex', 'age', 'mortalite']]
        .astype(dict(age = 'int'))
        .set_index(['sex', 'age'])
        .rename(columns = dict(mortalite = 'mortalite_{}'.format(year)))
        )


def get_mortality_after_imputation(period = 2010):
    data_by_sex = dict()
    for sex in ['male', 'female']:
        sexe_nbr = 0 if sex == 'male' else 1
        df = (pd.read_csv(os.path.join(assets_path, 'dependance_niveau.csv'), index_col = 0)
            .query('(period == @period) and (age >= 60)'))
        assert (df.query('dependance_niveau == -1')['total'] == 0).all()
        data_by_sex[sex] = (df
            .query('dependance_niveau != -1')
            .query('sexe == @sexe_nbr')
            .query('dependance_niveau != 5')
            .rename(columns = dict(dependance_niveau = 'initial_state'))
            .assign(sex = sex)
            .drop('sexe', axis = 1)
            )

    data = pd.concat(data_by_sex.values())

    mortalite_after_imputation = (data
        .merge(
            mortality_table[['sex', 'age', 'initial_state', 'mortality']],
            on = ['sex', 'age', 'initial_state'],
            how = 'inner',
            )
        .drop('period', axis = 1)
        .groupby(['sex', 'age'])[['total', 'mortality']].apply(lambda x: (
            (x.total * x.mortality).sum() / (x.total.sum() + (x.total.sum() == 0))
            ))
        )

    mortalite_after_imputation.name = 'mortalite_after_imputation'
    return mortalite_after_imputation


def plot_paquid_comparison(formula = None, age_max = 120):
    assert formula is not None
    transitions = get_transitions_from_formula(formula = formula)

    mortality_table = get_predicted_mortality_table(transitions = transitions).reset_index()

    filtered = get_filtered_paquid_data()
    filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)

#    filtered['age_group_10'] = 10 * (filtered.age / 10).apply(np.floor).astype('int')
#    filtered['age_group_5'] = 5 * ((filtered.age) / 5).apply(np.floor).astype('int')

    filtered['sex'] = 'male'
    filtered.loc[filtered.sexe == 2, 'sex'] = 'female'
    del filtered['sexe']

#    if sex in ['homme', 'male']:
#        sexe_nbr = 1
#    elif sex in ['femme', 'female']:
#        sexe_nbr = 2

#    test_df = filtered.query('(annee == 2003) & (initial_state != 5)').dropna()
#    test_df.initial_state.value_counts()
#
#    test_df.groupby(['age_group_5'])[['final_state']].apply(
#        lambda x: 1 - np.sqrt(1 - 1.0 * (x.final_state == 5).sum() / x.count())
#        )

    mortalite_1988 = extract_historical_mortality(year = 1998)
    mortalite_insee_2007 = get_mortality_from_insee_projection(year = 2007)

    profile = filtered.query('initial_state != 5').dropna()
    profile['age'] = profile.age.round()
    profile = (profile
        .merge(mortality_table, on =['sex', 'age', 'initial_state'], how = 'left')
        .groupby(['sex', 'age'])['mortality']
        .mean()
        .reset_index()
        .astype(dict(age = 'int'))
        .set_index(['sex', 'age'])
        )
    profile.name = 'mortality_from_paquid'

    mortalite_after_imputation = get_mortality_after_imputation(period = 2010)

    plot_data = (pd.concat(
        [profile, mortalite_insee_2007, mortalite_1988, mortalite_after_imputation],
        axis = 1,
        )
        .query('(age > 60) & (age < @age_max)')
        .reset_index()
        )
    axes = plot_data.groupby('sex').plot(x = 'age')
    for index in axes.index:
        axes[index].set_title("{} mortality".format(index))
        axes[index].get_figure().savefig('mortalite_{}.png'.format(index))

    plot_data['ratio'] = plot_data.eval('mortality / mortalite_insee_2007')

    axes2 = plot_data.groupby('sex').plot(x = 'age', y = 'ratio')

    for index in axes2.index:
        axes2[index].set_title(index)
        axes2[index].get_figure().savefig('ratio_mortalite_{}.png'.format(index))

    return plot_data


if __name__ == '__main__':

    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'

    # plot_data = plot_paquid_comparison(formula = formula, age_max = 95)
    period = 2010
    transitions = get_transitions_from_formula(formula = formula)
    result = build_mortality_calibrated_targets(transitions, period)