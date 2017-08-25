#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import numpy as np
import logging
import os
import pandas as pd


from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    assets_path,
    final_states_by_initial_state,
    get_transitions_from_formula,
    til_france_path,
    )

log = logging.getLogger(__name__)


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
    diff.name = 'error'

    assert (diff.abs().max() < 1e-10).all(), "error is too big: {} > 1e-10. Example: {}".format(
        diff.abs().max(), (dataframe
            .reset_index()
            .set_index(by)
            .loc[diff.abs().argmax(), ['final_state', probability]]
            .reset_index()
            .set_index(by + ['final_state'])
            )
        )


def build_mortality_calibrated_target(transitions = None, period = None):
    """
    Compute the calibrated mortality by sex, age and disability state (initial_state) for a given period
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
    mortality_calibrated_target = age_full[
        ['period', 'sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ].set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])

    assert_probabilities(
        dataframe = mortality_calibrated_target,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    return mortality_calibrated_target


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
    Calibrate mortality using the distribution of the disability states within population at a specific year
    for the given transition matrix
    Assuming the transition occur on a two-year period.
    """
    assert period is not None
    assert transitions is not None
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions)
    mortality_after_imputation = (
        get_mortality_after_imputation(mortality_table = predicted_mortality_table)
        .reset_index()
        .rename(columns = {'mortality_after_imputation': 'avg_mortality'})
        )

    projected_mortality = (get_insee_projected_mortality()
        .query('year == @period')
        .rename(columns = dict(year = 'period'))
        )

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
        transitions.reset_index().loc[transitions.reset_index()[
            ['sex', 'age', 'initial_state', 'final_state']].duplicated()]

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
    assert_probabilities(
        calibrated_transitions, by = ['sex', 'age', 'initial_state'], probability = 'calibrated_probability')
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


def add_projection_corrections(mortality_calibrated_target, mu = None, variant = None):
    regularize(
        transition_matrix_dataframe = mortality_calibrated_target,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

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

    assert not (correction_coefficient['correction_coefficient'].isnull().any())
    assert not (mortality_calibrated_target['calibrated_probability'].isnull().any()), \
        "There are {} NaN(s) in mortality_calibrated_target".format(
            mortality_calibrated_target['calibrated_probability'].isnull().sum())

    uncalibrated_probabilities = (mortality_calibrated_target.reset_index()[
        ['sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ]
        .merge(correction_coefficient)
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )

    assert not (uncalibrated_probabilities['calibrated_probability'].isnull().any()), \
        "There are {} NaN(s) in uncalibrated_probabilities".format(
            uncalibrated_probabilities['calibrated_probability'].isnull().sum())

    assert_probabilities(
        dataframe = uncalibrated_probabilities,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == 5')
        ).copy()
    assert (mortality.calibrated_probability * mortality.correction_coefficient < 1).all(), \
        "There are {} periodized_calibrated_probability >= 1".format(
            (mortality.calibrated_probability * mortality.correction_coefficient < 1).sum()
            )
    mortality['periodized_calibrated_probability'] = np.minimum(  # use minimum to avoid over corrections !
        mortality.calibrated_probability * mortality.correction_coefficient, 1 - 1e-5)

    assert (
        (mortality['periodized_calibrated_probability'] >= 0) &
        (mortality['periodized_calibrated_probability'] < 1)
        ).all()
    assert not (mortality['periodized_calibrated_probability'].isnull().any()), \
        "There are periodized_calibrated_probability NaNs in mortality"

    if mu is None:
        assert (mortality.calibrated_probability < 1).all()
        mortality.eval(
            'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
            inplace = True,
            )
        assert not mortality.cale_other_transitions.isnull().any(), "Some calibration coeffecients are NaNs"
        assert (mortality.cale_other_transitions > 0).all(), "Some calibration coeffecients are negative"

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
        # Ensures that the mortality is the projected one assuming no variation in elderly disability distribution
        assert not (other_transitions['periodized_calibrated_probability'].isnull().any()), \
            "There are {} NaN(s) in other_transitions".format(
                other_transitions['periodized_calibrated_probability'].isnull().sum())
        assert (
            (other_transitions['periodized_calibrated_probability'] >= 0) &
            (other_transitions['periodized_calibrated_probability'] <= 1)
            ).all(), "Erroneous periodized_calibrated_probability"
    else:
        assert variant is not None
        if variant == 0:
            other_transitions = _variant_0(
                mortality = mortality, mu = mu, uncalibrated_probabilities = uncalibrated_probabilities)
        elif variant == 1:
            other_transitions = _variant_1(
                mortality = mortality, mu = mu, uncalibrated_probabilities = uncalibrated_probabilities)
        else:
            raise NotImplementedError

    periodized_calibrated_transitions = (pd.concat([
        mortality.reset_index()[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        other_transitions[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        ]))

    # assert_probabilities(
    #     dataframe = periodized_calibrated_transitions,
    #     by = ['period', 'sex', 'age', 'initial_state'],
    #     probability = 'periodized_calibrated_probability',
    #     )
    return (periodized_calibrated_transitions
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )


def _variant_0(mortality = None, mu = None, uncalibrated_probabilities = None):
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
            to_initial_transitions[
                ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
                ],
            to_non_initial_transitions[
                ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
                ].fillna(0)
            ])

    return other_transitions


def _variant_1(mortality = None, mu = None, uncalibrated_probabilities = None):

    # Starting from autonomy
    # Staying autonomous
    mortality.eval(
        'survival_gain = - (periodized_calibrated_probability - calibrated_probability)',  # It is postive gain
        inplace = True,
        )
    survival_gain = mortality.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'survival_gain']
        ].copy()

    stay_autonomous = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state == 0)')
        .merge(survival_gain)
        .eval(
            'periodized_calibrated_probability = @mu * survival_gain + calibrated_probability',
            inplace = False,
            )
        )
    # From autonomy to disablement
    disabling_transitions_aggregate_probability = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state != 0) & (final_state != 5)')
        .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
        .reset_index()
        .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
        )
    become_disabled = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state != 0) & (final_state != 5)')
        .merge(disabling_transitions_aggregate_probability)
        .merge(survival_gain)
        .eval(
            'periodized_calibrated_probability =  calibrated_probability * (1 + (1 - @mu) * survival_gain / aggregate_calibrated_probability)',
            inplace = False,
            )
        )

    # For initial_states >= 1
    mortality.eval(
        'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
        inplace = True,
        )
    cale_other_transitions = mortality.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'cale_other_transitions']
        ].copy()
    other_transitions = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state != 0) & (final_state != 5)')
        .merge(cale_other_transitions)
        .eval(
            'periodized_calibrated_probability = calibrated_probability * cale_other_transitions',
            inplace = False,
            )
        )

    other_transitions = pd.concat([
        stay_autonomous,
        become_disabled,
        other_transitions,
        ])

    return other_transitions


def plot_periodized_mortality_calibrated_targets(periodized_mortality_calibrated_target):
    years = [2010, 2020, 20]
    periodized_mortality_calibrated_target.query


def regularize(transition_matrix_dataframe = None, by = None, probability = None, delta = 1e-5):
    assert transition_matrix_dataframe is not None
    assert by is not None
    assert probability is not None
    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    mortality_transitions = transition_matrix_dataframe.query('final_state == 5').copy()

    # by_without_initial_state = [by_value for by_value in by if by_value != 'initial_state']
    problematic_indices = (mortality_transitions[probability]
        .loc[mortality_transitions[probability] > (1 - delta)]
        .reset_index()
        .drop(['final_state', probability] , axis = 1)
        )
    count = (problematic_indices
        .merge(
            transition_matrix_dataframe.reset_index()[by + ['final_state']])
        .query('final_state != 5')
        .groupby(by)
        .count()
        )
    correction = delta / count.final_state.astype('float')
    correction.name = 'correction'
    corrected_transition_matrix = (transition_matrix_dataframe.reset_index()
        .merge(correction.reset_index())
        .eval(
            'calibrated_probability = (final_state == 5) * (1 - @delta) + (final_state != 5) * correction',
            inplace = False,
            )
        .drop('correction', axis = 1)
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        )
    transition_matrix_dataframe.update(corrected_transition_matrix)
    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    assert (transition_matrix_dataframe.query('final_state == 5')['calibrated_probability'] != 1).all()
    return transition_matrix_dataframe

if __name__ == '__main__':

    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1)**2) + I(((age - 80) * 0.1)**3)'
    period = 2010

    transitions = get_transitions_from_formula(formula = formula)
    mortality_calibrated_target = build_mortality_calibrated_target(transitions, period)
    assert_probabilities(
        dataframe = mortality_calibrated_target,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    periodized_target = add_projection_corrections(mortality_calibrated_target = mortality_calibrated_target, mu = None, variant = None)


    df = periodized_target.loc[(slice(2010, 2015), u'female', 89, 4, slice(0, 5)),]

    for period in range(2010, 2015):
        x = periodized_target.loc[(period, u'female', 89, 4, slice(0, 5)),]
        print x * 1e10
        print (x.sum() - 1) * 10e10

    BOUM
    for sex in ['male', 'female']:
        filename = os.path.join('/home/benjello/data/til/input/dependance_transition_{}.csv'.format(sex))
        periodized_target.to_csv('periodized_result.csv')

        if filename is not None:
            (periodized_target
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
