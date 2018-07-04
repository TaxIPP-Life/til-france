#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from slugify import slugify
import sys


from til_core.config import Config
from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    assets_path,
    final_states_by_initial_state,
    get_transitions_from_formula,
    til_france_path,
    )
from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )


def correct_transitions_for_mortality(transitions, dependance_initialisation = None, mu = None,
        variant = None, period = None):
    """
    Take a transition matrix = mortality_calibrated_target and correct transitions to match period's mortality target
    according to a scenario defined by variant and mu
    """
    assert dependance_initialisation is not None
    assert variant is None or variant in [0, 1, 2]
    assert period is not None
    if variant == 1:
        assert mu is not None
    delta = 1e-7
    regularize(
        transition_matrix_dataframe = transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        delta = delta)
    projected_mortality = get_insee_projected_mortality()

    target_mortality = (projected_mortality
        .query('year == @period')
        .rename(columns = {'mortality': 'target_mortality'})
        .reset_index()
        .drop('year', axis = 1)
        )

    actual_mortality = (dependance_initialisation
            .merge(
                transitions.reset_index().drop('period', axis = 1),
                on = ['age', 'sex', 'initial_state'])
            .eval('new_population = population * calibrated_probability', inplace = False)
            .drop(['initial_state', 'calibrated_probability', 'population'], axis = 1)
            .rename(columns = {'new_population': 'population'})
            .groupby(['age', 'sex', 'period', 'final_state'])['population'].sum()
            .reset_index()
            .rename(columns = {'final_state': 'initial_state'})
            .copy()
            )
    actual_mortality['part'] = (
        (
            actual_mortality / actual_mortality.groupby(['age', 'sex'])
            .transform(sum)
            )['population']
        .fillna(1)  # Kill all non dying
        )
    actual_mortality = (actual_mortality
        .query('initial_state == 5')[['age', 'sex', 'period', 'part']]
        .rename(columns = {'part': 'mortality', 'period': 'year'})
        )
    correction_coefficient = (target_mortality.reset_index()
        .merge(actual_mortality)
        .eval('correction_coefficient = target_mortality / mortality', inplace = False)
        .rename(columns = dict(year = 'period'))
        .drop(['mortality', 'target_mortality'], axis = 1)
        )

    assert not (correction_coefficient['correction_coefficient'].isnull().any())

    uncalibrated_probabilities = (transitions.reset_index()[
        ['sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ]
        .merge(correction_coefficient[['period', 'sex', 'age', 'correction_coefficient']])
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )

    assert not (uncalibrated_probabilities['calibrated_probability'].isnull().any()), \
        "There are {} NaN(s) in uncalibrated_probabilities".format(
            uncalibrated_probabilities['calibrated_probability'].isnull().sum())

    assert (
        uncalibrated_probabilities.query('final_state == 5')['calibrated_probability'] < 1
        ).all(), "There are {} 1's in calibrated_probability".format(
            (uncalibrated_probabilities['calibrated_probability'] < 1).sum(),
            )

    assert_probabilities(
        dataframe = uncalibrated_probabilities,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == 5')
        ).copy()

    mortality['periodized_calibrated_probability'] = np.minimum(  # use minimum to avoid over corrections !
        mortality.calibrated_probability * mortality.correction_coefficient, 1 - delta)

    assert (
        (mortality['periodized_calibrated_probability'] < 1)
        ).all(), "There are {} 1's in periodized_calibrated_probability".format(
            (mortality['periodized_calibrated_probability'] < 1).sum(),
            )

    assert not (mortality['periodized_calibrated_probability'].isnull().any()), \
        "There are calibrated_probability NaNs in mortality"

    if (variant is None) and (mu is None):
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
        elif variant == 2:
            other_transitions = _variant_2(
                mortality = mortality, uncalibrated_probabilities = uncalibrated_probabilities)
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

    assert_probabilities(
        dataframe = periodized_calibrated_transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'periodized_calibrated_probability',
        cut_off = delta,
        )
    return (periodized_calibrated_transitions
        .rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )


def add_projection_corrections(mortality_calibrated_target, mu = None, variant = None, period = None, periods = None):
    """
    Take a mortality calibrated target on a specific period = period and project if on periods (subsequent periods if None)
    """
    assert variant is None or variant in [0, 1]
    if variant is not None:
        assert mu is not None
    delta = 1e-7
    regularize(
        transition_matrix_dataframe = mortality_calibrated_target,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        delta = delta)

    if period is None:
        period = 2010
        log.info('period is not set: using 2010')
    projected_mortality = get_insee_projected_mortality()
    initial_mortality = (projected_mortality
        .query('year == @period')
        .rename(columns = {'mortality': 'initial_mortality'})
        .reset_index()
        .drop('year', axis = 1)
        )
    if periods is None:
        periods_query = 'year >= 2010'
    else:
        periods_query = 'year in {}'.format(periods)
        log.debug('periods_query: {}'.format(periods_query))

    correction_coefficient = (projected_mortality.reset_index()
        .query(periods_query)
        .merge(initial_mortality)
        .eval('correction_coefficient = mortality / initial_mortality', inplace = False)
        .rename(columns = dict(year = 'period'))
        .drop(['mortality', 'initial_mortality'], axis = 1)
        )

    assert not (correction_coefficient['correction_coefficient'].isnull().any())

    uncalibrated_probabilities = (mortality_calibrated_target.reset_index()[
        ['sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ]
        .merge(correction_coefficient[['period', 'sex', 'age', 'correction_coefficient']])
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )

    assert not (uncalibrated_probabilities['calibrated_probability'].isnull().any()), \
        "There are {} NaN(s) in uncalibrated_probabilities".format(
            uncalibrated_probabilities['calibrated_probability'].isnull().sum())

    assert (
        uncalibrated_probabilities.query('final_state == 5')['calibrated_probability'] < 1
        ).all(), "There are {} 1's in calibrated_probability".format(
            (uncalibrated_probabilities['calibrated_probability'] < 1).sum(),
            )

    assert_probabilities(
        dataframe = uncalibrated_probabilities,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == 5')
        ).copy()

    mortality['periodized_calibrated_probability'] = np.minimum(  # use minimum to avoid over corrections !
        mortality.calibrated_probability * mortality.correction_coefficient, 1 - delta)

    assert (
        (mortality['periodized_calibrated_probability'] < 1)
        ).all(), "There are {} 1's in periodized_calibrated_probability".format(
            (mortality['periodized_calibrated_probability'] < 1).sum(),
            )

    assert not (mortality['periodized_calibrated_probability'].isnull().any()), \
        "There are calibrated_probability NaNs in mortality"

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

    assert_probabilities(
        dataframe = periodized_calibrated_transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'periodized_calibrated_probability',
        cut_off = delta,
        )
    return (periodized_calibrated_transitions
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )


def _variant_0(mortality = None, mu = None, uncalibrated_probabilities = None):
    """
    Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
    """
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
    """
    Autonomy and disability states are treated differently
    Gain in survival probability feeds by a proportion of mu the autonomy initial_state and
    1 - mu the transition to the disability states.
    Other transition share proportionnally the survival probability gain
    """
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
    assert (stay_autonomous.periodized_calibrated_probability <= 1).all()
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


def _variant_2(mortality = None, uncalibrated_probabilities = None):
    """
    Autonomy and disability initial states are treated differently
    Gain in survival probability feeds the transiton from autonomy intial_state (0)
    to weeak state (1) and other transtions from the autonomy intial_state are unchanged
    Other transition share proportionnally the survival probability gain
    """
    # Starting from autonomy
    # Staying autonomous
    mortality.eval(
        'survival_gain = - (periodized_calibrated_probability - calibrated_probability)',  # It is postive gain
        inplace = True,
        )
    survival_gain = mortality.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'survival_gain']
        ].copy()

    become_weak = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state == 1)')
        .merge(survival_gain)
        .eval(
            # 'periodized_calibrated_probability = @mu * survival_gain + calibrated_probability',
            'periodized_calibrated_probability = survival_gain + calibrated_probability',
            inplace = False,
            )
        )
    assert (become_weak.periodized_calibrated_probability <= 1).all()
    # Other transitions are untouched
    from_0_to_0_4_transitions = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state != 1) & (final_state != 5)')
        .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
        .reset_index()
        .rename(columns = dict(calibrated_probability = 'periodized_calibrated_probability'))
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
        become_weak,
        from_0_to_0_4_transitions,
        other_transitions,
        ])
    return other_transitions


def assert_probabilities(dataframe = None, by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability', cut_off = 1e-9):
    assert dataframe is not None
    assert not (dataframe[probability] < 0).any(), dataframe.loc[dataframe[probability] < 0]
    assert not (dataframe[probability] > 1).any(), dataframe.loc[dataframe[probability] > 1]
    diff = (
        dataframe.reset_index().groupby(by)[probability].sum() - 1)
    diff.name = 'error'

    assert (diff.abs().max() < cut_off).all(), "error is too big: {} > 1e-10. Example: {}".format(
        diff.abs().max(), (dataframe
            .reset_index()
            .set_index(by)
            .loc[diff.abs().argmax(), ['final_state', probability]]
            .reset_index()
            .set_index(by + ['final_state'])
            )
        )


def build_mortality_calibrated_target(transitions = None, period = None, dependance_initialisation = None):
    """
    Compute the calibrated mortality by sex, age and disability state (initial_state) for a given period
    using data on the disability states distribution in the population at that period
    if dependance_initialisation = None
    TODO should be merged with build_mortality_calibrated_target_from_transitions
    """
    assert (transitions is not None) and (period is not None)
    calibrated_transitions = _get_calibrated_transitions(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        )

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
    if age_max < 120:
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
    else:
        age_full = pd.concat([
            pre_65_null_fill,
            calibrated_transitions.reset_index(),
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


def build_mortality_calibrated_target_from_formula(formula = None, period = None):
    assert period is not None
    assert formula is not None
    transitions = get_transitions_from_formula(formula = formula)
    return build_mortality_calibrated_target_from_transitions(
        transitions = transitions,
        period = period)


def build_mortality_calibrated_target_from_transitions(transitions = None, period = None, dependance_initialisation = None):
    assert period is not None
    assert transitions is not None
    mortality_calibrated_target = build_mortality_calibrated_target(
        transitions = transitions,
        period = period,
        dependance_initialisation = dependance_initialisation,
        )
    assert_probabilities(
        dataframe = mortality_calibrated_target,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )
    return mortality_calibrated_target


def export_calibrated_dependance_transition(projected_target = None, filename_prefix = None):
    assert projected_target is not None
    for sex in ['male', 'female']:
        if filename_prefix is None:
            config = Config()
            input_dir = config.get('til', 'input_dir')
            filename = os.path.join(input_dir, 'dependance_transition_{}.csv'.format(sex))
        else:
            filename = os.path.join('{}_{}.csv'.format(filename_prefix, sex))
        projected_target.to_csv('periodized_result.csv')

        if filename is not None:
            (projected_target
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
            log.info('Saving {}'.format(filename))


def get_mortality_after_imputation(mortality_table = None, dependance_initialisation = None):
    """
    Compute total mortality from mortality by dependance initial state given dependance_initialisation_male/female.csv
    present in til/input_dir or from dependance_initialisation data_frame if not None
    """
    assert mortality_table is not None
    if dependance_initialisation is not None:
        data = dependance_initialisation.rename(columns = {'population': 'total'})
    else:
        data_by_sex = dict()
        log.info("Work only for period = 2010")
        for sex in ['male', 'female']:
            sexe_nbr = 0 if sex == 'male' else 1
            sexe = 'homme' if sex == 'male' else 'femme'
            config = Config()
            input_dir = config.get('til', 'input_dir')
            filename = os.path.join(input_dir, 'dependance_initialisation_{}.csv'.format(sexe))
            log.info('Using {} for corecting mortality for dependance state '.format(filename))
            df = (pd.read_csv(filename, names = ['age', 0, 1, 2, 3, 4], skiprows = 1)
                .query('(age >= 65)')
                )
            df['age'] = df['age'].astype('int')

            df = (
                pd.melt(
                    df,
                    id_vars = ['age'],
                    value_vars = [0, 1, 2, 3, 4],
                    var_name = 'initial_state',
                    value_name = 'total'
                    )
                .sort_values(['age', 'initial_state'])
                .set_index(['age', 'initial_state'])
                )
            assert (df.query('initial_state == -1')['total'] == 0).all()
            data_by_sex[sex] = (df
                .assign(sex = sex)
                )
            data = pd.concat(data_by_sex.values()).reset_index()

    mortality_after_imputation = (data
        .merge(
            mortality_table.reset_index()[['sex', 'age', 'initial_state', 'mortality']],
            on = ['sex', 'age', 'initial_state'],
            how = 'inner',
            )
        .groupby(['sex', 'age'])[['total', 'mortality']].apply(lambda x: (
            (x.total * x.mortality).sum() / (x.total.sum() + (x.total.sum() == 0))
            ))
        )
    mortality_after_imputation.name = 'mortality_after_imputation'
    return mortality_after_imputation


def _compute_calibration_coefficient(age_min = 65, period = None, transitions = None, dependance_initialisation = None):
    """
    Calibrate mortality using the distribution of the disability states within population at a specific year
    for the given transition matrix and distribution of intiial_states
    Assuming the transition occur on a two-year period.
    """
    assert period is not None, "Mortality profile period is not set"
    assert transitions is not None
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions)
    mortality_after_imputation = (
        get_mortality_after_imputation(
            mortality_table = predicted_mortality_table,
            dependance_initialisation = dependance_initialisation,
            )
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


def _get_calibrated_transitions(period = None, transitions = None, dependance_initialisation = None):
    """
    Calibrate transitions to match mortality from a specified period
    """
    assert (period is not None) and (transitions is not None)

    # Add calibration_coeffcients for mortality
    calibration = _compute_calibration_coefficient(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        )

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
    if other_transitions.calibrated_probability.isnull().any():
        print 'COUCOU'
        print other_transitions.loc[other_transitions.calibrated_probability.isnull()]
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
    assert probability_name in transitions.columns, "{} not present in transitions colmns: {}".format(
        probability_name,
        transitions.columns
        )
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


def get_insee_projected_population():
    '''
    Get population data from INSEE projections
    '''

    from til_france.plot.population import get_insee_projection

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


def plot_projected_target(age_max = 100, age_min = 60, projected_target = None, sex = None,
        title = None, title_template = None, years = None, probability_name = 'periodized_calibrated_probability',
        initial_states = None, final_states = None, french = False, save = True, legend_title = None):

    assert projected_target is not None
    assert probability_name in projected_target.columns, '{} not found in columns: {}'.format(
        probability_name, projected_target.columns)
    assert (sex is None) or (sex in ['male', 'female'])
    assert years is not None
    if sex is None:
        sexes = ['male', 'female']
    elif sex in ['male', 'female']:
        sexes = [sex]

    grids = dict()
    for sex in sexes:

        if initial_states is not None:
            projected_target = projected_target.copy().query('initial_state in @initial_states')
        if final_states is not None:
            projected_target = projected_target.copy().query('initial_state in @initial_states')

        data = (projected_target
            .query('(age >= @age_min) and (age <= @age_max)')
            .query('(period in @years) and (sex == @sex)')
            .reset_index()
            .set_index(['period', 'age', 'initial_state', 'final_state'])
            [probability_name]
            .reset_index()
            .rename(columns= {probability_name: "probability"})
            )

        if not french:
            grid = sns.FacetGrid(data, row = "initial_state", col = "final_state", hue = "period", despine = True)
            # sns.set(font_scale=1.5, rc={'text.usetex' : Tr(''ue})
            # grid.set_titles(template = r"${row_name} \rightarrow {col_name}$")
            grid.set_titles(template = "{row_name} -> {col_name}")
            grid.map(plt.plot, 'age', 'probability')
            grid.add_legend(title = 'years')
            plt.subplots_adjust(top = 0.95)
            if title_template:
                title = title_template.format(sex)
            elif title is not None:
                fig_title = "{} (sex = {})".format(title, sex)
            else:
                fig_title = "sex = {}".format(sex)
        else:
            data.rename(
                columns = {
                    "initial_state": u"état initial",
                    "final_state": u"état final",
                    "period": u"Année",
                    "age": u"Âge",
                    "probability": u"Probabilité",
                    },
                inplace = True,
                )
            grid = sns.FacetGrid(data, row = u"état initial", col = u"état final", hue = u"Année", despine = True)
            # sns.set(font_scale=1.5, rc={'text.usetex' : Tr(''ue})
            # grid.set_titles(template = r"${row_name} \rightarrow {col_name}$")
            grid.set_titles(template = "{row_name} -> {col_name}")
            grid.map(plt.plot, u"Âge", u"Probabilité")
            if legend_title is None:
                legend_title = u"Années"
            grid.add_legend(title = legend_title)
            plt.subplots_adjust(top = 0.75)
            if title_template:
                title = title_template.format("Homme" if sex == 'male' else 'Femme')
                fig_title = title
            elif title is not None:
                fig_title = "{} (sexe = {})".format(title, "Homme" if sex == 'male' else 'Femme')
            else:
                fig_title = "Homme" if sex == 'male' else 'Femme'

        grid.fig.suptitle(fig_title)

        if initial_states is not None:
            fig_title += '_from_'
            fig_title += '_'.join(str(state) for state in initial_states)
        if final_states is not None:
            fig_title += '_to_'
            fig_title += '_'.join(str(state) for state in final_states)

        grids[sex] = grid
        if save:
            # if filename is None:
            filename = slugify(fig_title, separator = '_')
            config = Config()
            figures_directory = config.get('dependance', 'figures_directory')
            log.info("Saving matrix transitions graph to file {}".format(os.path.abspath(filename)))
            grid.savefig(os.path.join(figures_directory, filename + ".png"))
            grid.savefig(os.path.join(figures_directory, filename + ".pdf"))

    return grids if (len(grids) > 1) else grid


def regularize(transition_matrix_dataframe = None, by = None, probability = None, delta = None):
    assert transition_matrix_dataframe is not None
    assert by is not None
    assert probability is not None
    assert delta is not None
    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    mortality_transitions = transition_matrix_dataframe.query('final_state == 5').copy()

    # by_without_initial_state = [by_value for by_value in by if by_value != 'initial_state']
    problematic_indices = (mortality_transitions[probability]
        .loc[mortality_transitions[probability] >= (1 - delta)]
        .reset_index()
        .drop(['final_state', probability], axis = 1)
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
        .merge(correction.reset_index())  # inner merge
        .fillna(0)
        )
    corrected_transition_matrix.loc[
        (corrected_transition_matrix.final_state == 5),
        probability
        ] = (1 - delta)
    corrected_transition_matrix.loc[
        corrected_transition_matrix.final_state != 5,
        probability
        ] = corrected_transition_matrix.correction

    corrected_transition_matrix.set_index(by + ['final_state'], inplace = True)
    assert (
        (corrected_transition_matrix[probability] > 0) &
        (corrected_transition_matrix[probability] < 1)
        ).all(), "There are {} 0's and {} 1's in corrected_transition_matrix[{}]".format(
            (corrected_transition_matrix[probability] > 0).sum(),
            (corrected_transition_matrix[probability] < 1).sum(),
            probability
            )
    transition_matrix_dataframe.update(corrected_transition_matrix)

    assert (
        transition_matrix_dataframe.query('final_state == 5')[probability] < 1
        ).all(), "There are {} 1's in transition_matrix_dataframe.{}".format(
            (transition_matrix_dataframe.query('final_state == 5')[probability] < 1).sum(),
            probability,
            )

    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    return transition_matrix_dataframe


def produce_simple_initialisation():
    """
    Simple function to produce standard initialisation file.
    TODO: move somewhere else
    """
    config = Config()
    input_dir = config.get('til', 'input_dir')
    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1)**2) + I(((age - 80) * 0.1)**3)'
    period = 2010
    mortality_calibrated_target = build_mortality_calibrated_target_from_formula(
        formula = formula, period = period)
    for mu in [0, 0.5, 1]:
        variant = 1
        title = "mu = {}, variant = {}".format(mu, variant)
        projected_target = add_projection_corrections(
            mortality_calibrated_target = mortality_calibrated_target, mu = mu, variant = variant)
        years = [2010, 2020, 2030, 2040, 2050]
        plot_projected_target(projected_target = projected_target, years = years, title = title)
        filename_prefix = os.path.join(
            input_dir,
            'dependance_transition_mu_{}_variant_{}'.format(mu, variant)
            )
        export_calibrated_dependance_transition(projected_target, filename_prefix)
