#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import numpy as np
import logging
import os
import pandas as pd


log = logging.getLogger(__name__)


from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    assets_path,
    final_states_by_initial_state,
    get_transitions_from_formula,
    til_france_path,
    )


life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )


def assert_probabilities(dataframe = None, by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability'):
    assert dataframe is not None
    assert not (dataframe[probability] < 0).any(), dataframe.loc[dataframe[probability] < 0]
    assert not (dataframe[probability] > 1).any(), dataframe.loc[dataframe[probability] > 1]
    diff = (
        dataframe.reset_index().groupby(by)[probability].sum() - 1)
    assert (diff.abs().max() < 1e-10).all(), "error is too big: {} > 1e-10. Example: {}".format(
        diff.abs().max(), diff.loc[(diff.abs() >= 1e-10)])


def build_mortality_calibrated_targets(transitions = None, period = None):
    """
    Compute the calibrated mortality  by sex, age and disability state (initial_state) for a given period
    using data on the disability states distribution in the population at that period
    """
    assert (transitions is not None) and (period is not None)
    calibrated_transitions = _get_calibrated_transitions(period = period, transitions = transitions)

    null_fill_by_year = calibrated_transitions.reset_index().query('age == 65').copy()
    null_fill_by_year['calibrated_probability'] = 0

    # Less than 65 years old _> no correction
    pre_65_null_fill = pd.concat([
        null_fill_by_year.assign(age = i).copy()
        for i in range(0, 65)
        ]).reset_index(drop = True)

    pre_65_null_fill.loc[
        pre_65_null_fill.initial_state == pre_65_null_fill.final_state,
        'calibrated_probability'
        ] = 1

    assert_probabilities(
        dataframe = pre_65_null_fill,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

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

    assert_probabilities(
        dataframe = elder_null_fill,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    age_full = pd.concat([
        pre_65_null_fill,
        calibrated_transitions.reset_index(),
        elder_null_fill
        ]).reset_index(drop = True)

    age_full['period'] = period
    result = age_full[
        ['period', 'sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ].set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])

    assert_probabilities(
        dataframe = result,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    return result


def get_mortality_after_imputation(period = 2010, mortality_table = None):
    assert mortality_table is not None
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

    mortality_after_imputation = (data
        .merge(
            mortality_table.reset_index()[['sex', 'age', 'initial_state', 'mortality']],
            on = ['sex', 'age', 'initial_state'],
            how = 'inner',
            )
        .drop('period', axis = 1)
        .groupby(['sex', 'age'])[['total', 'mortality']].apply(lambda x: (
            (x.total * x.mortality).sum() / (x.total.sum() + (x.total.sum() == 0))
            ))
        )

    mortality_after_imputation.name = 'mortality_after_imputation'
    return mortality_after_imputation


def _get_calibration(age_min = 65, period = None, transitions = None):
    """
    Calibrate mortality using the distribution of the disability states within population at period = period
    for the given transitions
    This assumes the transition occur on a two-year period.
    """
    assert period is not None
    assert transitions is not None
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions)
    mortality_after_imputation = (
        get_mortality_after_imputation(mortality_table = predicted_mortality_table)
        .reset_index()
        .rename(columns = {'mortality_after_imputation': 'avg_mortality'})
        )

    print mortality_after_imputation.head()
    projected_mortality = (get_insee_projected_mortality()
        .query('year == @period')
        .rename(columns = dict(year = 'period'))
        )
    print projected_mortality.head()

    model_to_target = (mortality_after_imputation
        .merge(
            projected_mortality.reset_index(),
            on = ['sex', 'age'],
            )
        .eval('cale_mortality_1_year = mortality / avg_mortality', inplace = False)
        .eval('mortalite_2_year = 1 - (1 - mortality) ** 2', inplace = False)
        .eval('avg_mortality_2_year = 1 - (1 - avg_mortality) ** 2', inplace = False)
        .eval('cale_mortality_2_year = mortalite_2_year / avg_mortality_2_year', inplace = False)
        )

    print model_to_target.head()
    return model_to_target


def _get_calibrated_transitions(period = None, transitions = None):
    """

    """
    assert (period is not None) and (transitions is not None)

    # Add calibration_coeffcients for mortality
    calibration = _get_calibration(period = period, transitions = transitions)

    assert not calibration.reset_index()[['sex', 'age']].duplicated().any(), \
        calibration.reset_index().loc[calibration.reset_index()[['sex', 'age']].duplicated()]

    # Calibrate mortality
    assert not transitions.reset_index()[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
        transitions.reset_index().loc[transitions.reset_index()[['sex', 'age', 'initial_state', 'final_state']].duplicated()]

    mortality = (transitions
        .reset_index()
        .query('final_state == 5')
        .merge(
            calibration[['sex', 'age', 'cale_mortality_2_year']].copy(),
            on = ['sex', 'age'],
            )
        )
    mortality['calibrated_probability'] = np.minimum(
        mortality.probability * mortality.cale_mortality_2_year, 1)  # Avoid over corrections !
    mortality.eval(
        'cale_other_transitions = (1 - calibrated_probability) / (1 - probability)',
        inplace = True,
        )
    assert not mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
        mortality.loc[mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated()]
    # Calibrate other transitions
    cale_other_transitions = mortality[['sex', 'age', 'initial_state', 'cale_other_transitions']].copy()
    other_transitions = (transitions
        .reset_index()
        .query('final_state != 5')
        .merge(
            cale_other_transitions,
            on = ['sex', 'age', 'initial_state'],
            )
        .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
        )
    calibrated_transitions = pd.concat(
        [
            mortality.set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index(),
            other_transitions.set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index()
            ]
        ).sort_index()

    # Verification
    assert_probabilities(calibrated_transitions, by = ['sex', 'age', 'initial_state'], probability = 'calibrated_probability')
    return calibrated_transitions['calibrated_probability']


def get_historical_mortality(rebuild = False):
    mortality_store_path = os.path.join(assets_path, 'historical_mortality.h5')
    sex_historical_mortalities = []
    if os.path.exists(mortality_store_path) and (not rebuild):
        historical_mortality = pd.read_hdf(mortality_store_path, key = 'historical_mortality')
    else:
        log.info('Rebuilding historical_mortality.h5')
        for sex in ['male', 'female']:
            sex_historical_mortality = (
                pd.read_excel(life_table_path, sheetname = 'france-{}'.format(sex))[['Year', 'Age', 'qx']]
                .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
                .replace(dict(age = {'110+': '110'}))
                )
            sex_historical_mortality['age'] = sex_historical_mortality['age'].astype('int')
            sex_historical_mortality['sex'] = sex
            sex_historical_mortalities.append(sex_historical_mortality)

        historical_mortality = pd.concat(sex_historical_mortalities)
        historical_mortality.to_hdf(mortality_store_path, key = 'historical_mortality')
    return historical_mortality


def get_predicted_mortality_table(transitions = None, save = False, probability_name = 'probability'):
    assert transitions is not None
    mortality_table = (transitions
        .query('final_state == 5')
        .copy()
        .assign(
            mortality = lambda x: (1 - np.sqrt(1 - x[probability_name]))
            )
        )
    if save:
        mortality_table.to_csv('predicted_mortality_table.csv')
    return mortality_table


def get_insee_projected_mortality():
    '''
    Get mortality data from INSEE projections
    '''
    data_path = os.path.join(til_france_path, 'param', 'demo')

    sheetname_by_sex = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_sex = dict(
        (
            sex,
            pd.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheetname = sheetname, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for sex, sheetname in sheetname_by_sex.iteritems()
        )

    for df in mortality_by_sex.values():
        del df[u"Âge atteint dans l'année"]
        df.index.name = 'age'

    mortality = None
    for sex in ['male', 'female']:
        mortality_sex = ((mortality_by_sex[sex] / 1e4)
            .reset_index()
            )
        mortality_sex = pd.melt(
            mortality_sex,
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'mortality'
            )
        mortality_sex['sex'] = sex
        mortality_sex.rename(columns = dict(annee = 'year'), inplace = True)
        mortality = pd.concat([mortality, mortality_sex])

    return mortality.set_index(['sex', 'age', 'year'])


def add_projection_corrections(result, mu = None):
    projected_mortality = get_insee_projected_mortality()
    initial_mortality = (projected_mortality
        .query('year == 2010')
        .rename(columns = {'mortality': 'initial_mortality'})
        .reset_index()
        .drop('year', axis = 1)
        )
    correction_coefficient = (projected_mortality.reset_index()
        .query('year >= 2010')
        .merge(initial_mortality)
        .eval('correction_coefficient = mortality / initial_mortality', inplace = False)
        .rename(columns = dict(year = 'period'))
        .drop(['mortality', 'initial_mortality'], axis = 1)
        )
    result = pd.read_csv('result.csv')

    uncalibrated_probabilities = (result[
        ['period', 'sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ]
        .merge(correction_coefficient)
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == 5')
        ).copy()
    mortality['periodized_calibrated_probability'] = np.minimum(  # use minimum to avoid over corrections !
        mortality.calibrated_probability * mortality.correction_coefficient, 1)

    if mu is None:
        mortality.eval(
            'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
            inplace = True,
            )
        cale_other_transitions = mortality.reset_index()[
            ['period', 'sex', 'age', 'initial_state', 'cale_other_transitions']
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
        mortality.eval(
            'delta_initial = - @mu * (periodized_calibrated_probability - calibrated_probability)',
            inplace = True,
            )

        mortality.eval(
            'delta_non_initial = - (1 - @mu) * (periodized_calibrated_probability - calibrated_probability)',
            inplace = True,
            )

        for initial_state, final_states in final_states_by_initial_state.iteritems():
            if 5 not in final_states:
                continue
            #
            delta_initial_transitions = mortality.reset_index()[
                ['period', 'sex', 'age', 'initial_state', 'delta_initial']
                ].copy()

            to_initial_transitions = (uncalibrated_probabilities
                .reset_index()
                .query('(final_state == @initial_state) and (final_state != 5)')
                .merge(delta_initial_transitions)
                )
            to_initial_transitions['periodized_calibrated_probability'] = np.maximum(
                to_initial_transitions.calibrated_probability + to_initial_transitions.delta_initial, 0)

            delta_non_initial_transitions = mortality.reset_index()[
                ['period', 'sex', 'age', 'initial_state', 'delta_non_initial']
                ].copy()

            non_initial_transitions_aggregate_probability = (uncalibrated_probabilities
                .reset_index()
                .query('(final_state != @initial_state) and (final_state != 5)')
                .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
                .reset_index()
                .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
                )

            to_non_initial_transitions = (uncalibrated_probabilities
                .reset_index()
                .query('(final_state != @initial_state) and (final_state != 5)')
                .merge(non_initial_transitions_aggregate_probability.reset_index())
                .merge(delta_non_initial_transitions)
                .eval(
                    'periodized_calibrated_probability = calibrated_probability * (1 + delta_non_initial / aggregate_calibrated_probability)',
                    inplace = False,
                    )
                )

            other_transitions = pd.concat([
                to_initial_transitions[['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']],
                to_non_initial_transitions[['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']].fillna(0)
                ])

    periodized_calibrated_transitions = (pd.concat([
        mortality.reset_index()[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        other_transitions[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        ]))

    assert_probabilities(
        dataframe = periodized_calibrated_transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'periodized_calibrated_probability',
        )
    return (periodized_calibrated_transitions
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )


if __name__ == '__main__':

    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1)**2) + I(((age - 80) * 0.1)**3)'
    period = 2010

    transitions = get_transitions_from_formula(formula = formula)
    result = build_mortality_calibrated_targets(transitions, period)
    assert_probabilities(
        dataframe = result,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    periodized_result = add_projection_corrections(result = result, mu = .5)

    for sex in ['male', 'female']:
        filename = os.path.join('/home/benjello/data/til/input/dependance_transition_{}.csv'.format(sex))
        periodized_result.to_csv('periodized_result.csv')

        if filename is not None:
            print(periodized_result
                .query('sex == @sex')
                .reset_index()
                .drop('sex', axis = 1)
                .set_index(['period', 'age', 'initial_state', 'final_state'])
                .unstack()
                .fillna(0)
                )
            (periodized_result
                .query('sex == @sex')
                .reset_index()
                .drop('sex', axis = 1)
                .set_index(['period', 'age', 'initial_state', 'final_state'])
                .unstack()
                .fillna(0)
                .to_csv(filename, header = False)
                )
            # Add liam2 compatible header
            verbatim_header = '''period,age,initial_state,final_state,,,,,
,,,0,1,2,3,4,5
'''
            with file(filename, 'r') as original:
                data = original.read()
            with file(filename, 'w') as modified:
                modified.write(verbatim_header + data)
