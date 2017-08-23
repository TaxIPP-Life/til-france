#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import pandas as pd


from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    get_clean_paquid,
    )

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    build_mortality_calibrated_targets,
    get_historical_mortality,
    get_mortality_after_imputation,
    get_predicted_mortality_table,
    get_transitions_from_formula,
    )


from til_france.targets.population import build_mortality_rates


def extract_historical_mortality(year = None):
    return (get_historical_mortality()
        .query('(annee == @year)')
        .query('age <= 110')[['sex', 'age', 'mortalite']]
        .astype(dict(age = 'int'))
        .set_index(['sex', 'age'])
        .rename(columns = dict(mortalite = 'mortalite_{}'.format(year)))
        )


def get_calibrated_mortality_after_imputation(transitions, period):
    calibrated_transitions = build_mortality_calibrated_targets(
        transitions = transitions,
        period = period,
        )

    calibrated_mortality_table = get_predicted_mortality_table(
        transitions = calibrated_transitions,
        probability_name = 'calibrated_probability',
        )

    calibrated_mortality_after_imputation = get_mortality_after_imputation(
        period = period,
        mortality_table = calibrated_mortality_table,
        )

    calibrated_mortality_after_imputation.name = 'calibrated_mortality_after_imputation'

    return calibrated_mortality_after_imputation


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


def plot_paquid_comparison(formula = None, age_max = 120):
    """Plot various mortality curves by age  and sex
    mortalite_1988: historical mortality values
    mortalite_insee_2007: prjected mortality values
    mortalite
    """
    assert formula is not None
    transitions = get_transitions_from_formula(formula = formula)
    mortality_table = get_predicted_mortality_table(transitions = transitions)

    paquid = get_clean_paquid()
    paquid['final_state'] = paquid.groupby('numero')['initial_state'].shift(-1)

    paquid['sex'] = 'male'
    paquid.loc[filtered.sexe == 2, 'sex'] = 'female'
    del paquid['sexe']

    mortalite_1988 = extract_historical_mortality(year = 1988)
    mortalite_insee_2007 = get_mortality_from_insee_projection(year = 2007)

    profile = paquid.query('initial_state != 5').dropna()
    profile['age'] = profile.age.round()
    profile = (profile
        .merge(mortality_table.reset_index(), on =['sex', 'age', 'initial_state'], how = 'left')
        .groupby(['sex', 'age'])['mortality']
        .mean()
        .reset_index()
        .astype(dict(age = 'int'))
        .set_index(['sex', 'age'])
        )
    profile.name = 'mortality_from_paquid'

    period = 2010
    mortality_after_imputation = get_mortality_after_imputation(period = period, mortality_table = mortality_table)
    calibrated_mortality_after_imputation = get_calibrated_mortality_after_imputation(
        transitions = transitions, period = period)

    plot_data = (pd.concat(
        [
            profile,
            mortalite_insee_2007,
            mortalite_1988,
            mortality_after_imputation,
            calibrated_mortality_after_imputation,
            ],
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

    plot_data = plot_paquid_comparison(formula = formula, age_max = 95)

#    period = 2010
#    transitions = get_transitions_from_formula(formula = formula)
#    mortality_table = get_predicted_mortality_table(transitions = transitions)
#
#    calibrated_mortality_after_imputation.reset_index().query('(age >= 65) and (age <= 95)').groupby('sex').plot(
#        x = 'age', y = 'calibrated_mortality_after_imputation')
