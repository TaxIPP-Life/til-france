#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import numpy as np
import os
import pandas as pd


from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    til_france_path,
    )


life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )


def get_calibration(age_min = 65, period = None, transitions = None):
    assert period is not None
    assert transitions is not None
    predicted_mortality = get_predicted_mortality_table(formula = formula)

    population_sample = (pd.read_csv(os.path.join(assets_path, 'dependance_niveau.csv'), index_col = 0)
        .query('(period == @period) and (age >= @age_min)')
        .query('dependance_niveau != -1')
        .query('dependance_niveau != 5')
        )
    population_sample.loc[
        population_sample.sexe == 0,
        'sex'
        ] = 'male'
    population_sample.loc[
        population_sample.sexe == 1,
        'sex'
        ] = 'female'

    mortalite_after_imputation = (population_sample.merge(
        predicted_mortality.reset_index().rename(columns = dict(initial_state = 'dependance_niveau')),
        how = 'inner')
        )
    #

    global_mortalite_after_imputation = (mortalite_after_imputation
        .groupby(['age'])[['total', 'mortality']].apply(
            lambda x: (x.total * x.mortality).sum() / (x.total.sum() + (x.total.sum() == 0)))
        .reset_index()
        .rename(columns = {0: 'avg_mortality'})
        )
    mortalite_by_sex = get_projection_mortalite_by_sex()


    mortalite_reelle = (mortalite_by_sex[sex]
        .query('annee == @period')
        .reset_index()
        .loc[:109, ['age', 'mortalite']]
        .astype(dict(age = 'int'))
        .rename(columns = dict(annee = 'period'))
        )

    model_to_target = (global_mortalite_after_imputation.merge(mortalite_reelle)
        .eval('cale_mortality_1_year = mortalite / avg_mortality', inplace = False)
        .eval('mortalite_2_year = 1 - (1 - mortalite) ** 2', inplace = False)
        .eval('avg_mortality_2_year = 1 - (1 - avg_mortality) ** 2', inplace = False)
        .eval('cale_mortality_2_year = mortalite_2_year / avg_mortality_2_year', inplace = False)
        )
    return model_to_target


def get_calibrated_transitions(period = None, transitions = None):
    assert period is not None

    # Add calibration_coeffcients
    calibration = get_calibration(period = period, transitions = transitions)

    # Calibrate mortality
    mortality = (transitions
        .reset_index()
        .query('final_state == 5')
        .merge(calibration[['age', 'cale_mortality_2_year']], on = 'age')
        )
    mortality['calibrated_probability'] = np.minimum(
        mortality.probability * mortality.cale_mortality_2_year, 1)  # Avoid over corrections !
    mortality.eval(
        'cale_other_transitions = (1 - calibrated_probability) / (1 - probability)',
        inplace = True,
        )
    # Calibrate other transitions
    cale_other_transitions = mortality[['sex', 'age', 'initial_state', 'cale_other_transitions']].copy()
    other_transitions = (transitions
        .reset_index()
        .query('final_state != 5')
        .merge(cale_other_transitions)
        .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
        )
    calibrated_transitions = (pd.concat([mortality, other_transitions])
        .set_index(['sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )
    # Verification
    diff = (
        calibrated_transitions.reset_index().groupby(['sex', 'age', 'initial_state'])['calibrated_probability'].sum() - 1)
    assert (diff.abs().max() < 1e-10).all(), "error is too big: {} > 1e-10".format(diff.abs().max())

    return calibrated_transitions['calibrated_probability']


def get_historical_mortality(rebuild = False):
    mortality_store_path = os.path.join(assets_path, 'historical_mortality.h5')
    historical_mortality = None
    if os.path.exists(mortality_store_path) and not rebuild:
        historical_mortality = pd.read_hdf(mortality_store_path, key = 'historical_mortality')
    else:
        for sex in ['male', 'female']:
            sex_historical_mortality = (
                pd.read_excel(life_table_path, sheetname = 'france-{}'.format(sex))[['Year', 'Age', 'qx']]
                .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
                .replace(dict(age = {'110+': '110'}))
                )
            sex_historical_mortality['age'] = sex_historical_mortality['age'].astype('int')
            historical_mortality = pd.concat([historical_mortality, sex_historical_mortality])
        historical_mortality.to_hdf(mortality_store_path, key = 'historical_mortality')
    return historical_mortality


def get_predicted_mortality_table(formula = None, save = True):
    assert formula is not None
    mortality_table = (get_transitions_from_formula(formula = formula)
        .query('final_state == 5')
        .copy()
        .assign(
            mortality = lambda x: (1 - np.sqrt(1 - x.probability))
            )
        )
    if save:
        mortality_table.to_csv('predicted_mortality_table.csv')
    return mortality_table


def get_projection_mortalite_by_sex():
    # Data from INSEE projections
    data_path = os.path.join(til_france_path, 'param', 'demo')

    sheetname_by_gender = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_gender = dict(
        (
            gender,
            pd.read_excel(
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
        df.index.name = 'age'

    mortalite_by_sex = dict()
    for gender in ['male', 'female']:
        mortalite_by_sex[gender] = mortality_by_gender[gender] / 1e4
        mortalite_by_sex[gender] = mortalite_by_sex[gender].reset_index()
        mortalite_by_sex[gender] = pd.melt(
            mortalite_by_sex[gender],
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'mortalite'
            )

    return mortalite_by_sex


def add_projection_corrections(sex, result, mu = None):
    projection_mortalite = get_projection_mortalite_by_sex()[sex]
    initial_mortalite = (projection_mortalite
        .query('annee == 2010')
        .rename(columns = {'mortalite': 'initial_mortalite'})
        .drop('annee', axis = 1)
        )
    correction_coefficient = (projection_mortalite
        .query('annee >= 2010')
        .merge(initial_mortalite.reset_index())
        .eval('correction_coefficient = mortalite / initial_mortalite', inplace = False)
        .rename(columns = dict(annee = 'period'))
        .drop(['mortalite', 'initial_mortalite'], axis = 1)
        )

    result = pd.read_csv('result.csv')
    uncalibrated_probabilities = (result[['age', 'initial_state', 'final_state', 'calibrated_probability']]
        .merge(correction_coefficient)
        .set_index(['period', 'age', 'initial_state'])
        .sort_index()
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == 5')
        ).copy()
    mortality['periodized_calibrated_probability'] = np.minimum(
        mortality.calibrated_probability * mortality.correction_coefficient, 1)  # Avoid over corrections !

    if mu is None:
        mortality.eval(
            'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
            inplace = True,
            )
        cale_other_transitions = mortality.reset_index()[
            ['period', 'age', 'initial_state', 'cale_other_transitions']
            ].copy()
        other_transitions = (uncalibrated_probabilities
            .reset_index()
            .query('final_state != 5')
            .merge(cale_other_transitions)
            .eval(
                'periodized_calibrated_probability = calibrated_probability * cale_other_transitions',
                inplace = False,
                )
            )
    else:
        raise('mu should be None')
#        mortality.eval(
#            'delta_initial = - @mu * (periodized_calibrated_probability - calibrated_probability)',
#            inplace = True,
#            )
#
#        mortality.eval(
#            'delta_non_initial = - (1 - @mu) * (periodized_calibrated_probability - calibrated_probability)',
#            inplace = True,
#            )
#
#        for initial_state, final_states in final_states_by_initial_state.iteritems():
#            if 5 not in final_states:
#                continue
#            #
#            initial_state = 1
#            delta_initial_transitions = mortality.reset_index()[
#                ['period', 'age', 'initial_state', 'delta_initial']
#                ].copy()
#
#            to_initial_transitions = (uncalibrated_probabilities
#                .reset_index()
#                .query('(final_state == @initial_state) and (final_state != 5)')
#                .merge(delta_initial_transitions)
#                .eval(
#                    'periodized_calibrated_probability = calibrated_probability + delta_initial',  # Cannot be bigger than 1
#                    inplace = False,
#                    )
#                )
#
#            non_initial_transitions_aggregate_probability = (uncalibrated_probabilities
#                .reset_index()
#                .query('(final_state != @initial_state) and (final_state != 5)')
#                .groupby(['period', 'age', 'initial_state'])['calibrated_probability'].sum()
#                )
#
#            to_non_initial_transitions = (uncalibrated_probabilities
#                .reset_index()
#                .query('(final_state != @initial_state) and (final_state != 5)')
#                .eval(
#                    'periodized_calibrated_probability = calibrated_probability + delta_non_initial_transitions',
#                    inplace = False,
#                    )
#                .merge(delta_non_initial_transitions)
#                .eval(
#                    'periodized_calibrated_probability = calibrated_probability + delta_non_initial_transitions',
#                    inplace = False,
#                    )
#                )

    periodized_calibrated_transitions = (pd.concat([
        mortality.reset_index()[
            ['period', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        other_transitions[
            ['period', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        ]))
    return periodized_calibrated_transitions.set_index(['period', 'age', 'initial_state', 'final_state'])


if __name__ == '__main__':

    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1)**2) + I(((age - 80) * 0.1)**3)'
    period = 2010

    sex = 'female'
    df = get_predicted_mortality_table(formula = formula)

    for sex in ['male', 'female']:
        filename = os.path.join('/home/benjello/data/til/input/dependance_transition_{}.csv'.format(sex))
        #    def export_calibrated_transitions_to_liam(formula = None, period = None, sexe = None, filename = None):
        transitions = get_transitions_from_formula(formula = formula)
        calibrated_transitions = get_calibrated_transitions(period = period, transitions = transitions)

        null_fill_by_year = calibrated_transitions.reset_index().query('age == 65').copy()
        null_fill_by_year['calibrated_probability'] = 0

        # Less than 65 years old _> no correction
        pre_65_null_fill = pd.concat([
            null_fill_by_year.assign(age = i).copy()
            for i in range(0, 65)
            ]).reset_index(drop = True)

        pre_65_null_fill.loc[
            pre_65_null_fill.final_state == 0,
            'calibrated_probability'
            ] = 1

        # More than 65 years old
        age_max = calibrated_transitions.index.get_level_values('age').max()
        elder_null_fill = pd.concat([
            null_fill_by_year.assign(age = i).copy()
            for i in range(age_max + 1, 121)
            ]).reset_index(drop = True)

        elder_null_fill.loc[
            (elder_null_fill.age > age_max) & (elder_null_fill.final_state == 5),
            'calibrated_probability'
            ] = 1

        age_full = pd.concat([
            pre_65_null_fill,
            calibrated_transitions.reset_index(),
            elder_null_fill
            ]).reset_index(drop = True)

        age_full['period'] = period
        result = age_full[
            ['period', 'age', 'initial_state', 'final_state', 'calibrated_probability']
            ].set_index(['period', 'age', 'initial_state', 'final_state'])

        assert not (result.calibrated_probability < 0).any(), result.loc[result.calibrated_probability < 0]
        assert not (result.calibrated_probability > 1).any(), result.loc[result.calibrated_probability > 1]
        result.to_csv('result.csv')

        periodized_result = add_projection_corrections(sex = sex, result = result, mu = None)
        periodized_result.to_csv('periodized_result.csv')

        if filename is not None:
            periodized_result.unstack().fillna(0).to_csv(filename, header = False)
            # Add liam2 compatible header
            verbatim_header = '''period,age,initial_state,final_state,,,,,
,,,0,1,2,3,4,5
'''
            with file(filename, 'r') as original:
                data = original.read()
            with file(filename, 'w') as modified:
                modified.write(verbatim_header + data)

    # return age_full[['period', 'age', 'initial_state', 'final_state']]

    bim