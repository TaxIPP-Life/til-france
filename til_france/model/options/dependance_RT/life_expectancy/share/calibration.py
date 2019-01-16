#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import logging

import numpy as np
import os
import pandas as pd
import sys


from til_core.config import Config
from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
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


def correct_transitions_for_mortality(transitions, dependance_initialisation = None, mu = None, period = None,
        survival_gain_cast = None, previous_mortality = None):
    """
    Take a transition matrix = mortality_calibrated_target and correct transitions to match period's mortality target
    according to a scenario defined by survival_gain_cast and mu
    """
    death_state = 4
    assert dependance_initialisation is not None
    admissible_survival_gain_cast = [
        "homogeneous", "initial_vs_others", 'autonomy_vs_disability',
        ]
    assert survival_gain_cast in admissible_survival_gain_cast, \
        "suvival_gain_cast should one of the following values {}".format(admissible_survival_gain_cast)
    assert period is not None
    delta = 1e-7
    regularize(
        transition_matrix_dataframe = transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        delta = delta
        )

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
            actual_mortality / actual_mortality.groupby(['age', 'sex']).transform(sum)
            )['population']
        .fillna(1)  # Kill all non dying
        )
    actual_mortality = (actual_mortality
        .query('initial_state == @death_state')[['age', 'sex', 'period', 'part']]
        .rename(columns = {'part': 'mortality', 'period': 'year'})
        )
    correction_coefficient = (target_mortality.reset_index()
        .merge(actual_mortality)
        .eval('correction_coefficient = (1 - (1 - target_mortality) ** 2 ) / mortality', inplace = False)
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
        uncalibrated_probabilities.query('final_state == @death_state')['calibrated_probability'] < 1
        ).all(), "There are {} 1's in calibrated_probability".format(
            (uncalibrated_probabilities['calibrated_probability'] < 1).sum(),
            )

    assert_probabilities(
        dataframe = uncalibrated_probabilities,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == @death_state')
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

    if survival_gain_cast == "homogeneous":
        # Gain in survival probability are dispatched to the other states respecting the orginal odds ratio
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
            .query('final_state != @death_state')
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

    elif survival_gain_cast == "initial_vs_others":
        assert mu is not None
        # Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
        other_transitions = initial_vs_others(
            period = period,
            target_mortality = mortality,
            # previous_mortality = previous_mortality,
            mu = mu,
            uncalibrated_probabilities = uncalibrated_probabilities
            )
    elif survival_gain_cast == 'autonomy_vs_disability':
        assert mu is not None
        other_transitions = autonomy_vs_disability(
            mortality = mortality,
            mu = mu,
            uncalibrated_probabilities = uncalibrated_probabilities,
            )
    else:
        raise NotImplementedError

    mortality_recomputed = (other_transitions
        .query('final_state != @death_state')
        .groupby(
            ['period', 'sex', 'age', 'initial_state']
            )['periodized_calibrated_probability'].sum()
        .reset_index()
        )

    mortality_recomputed['periodized_calibrated_probability'] = 1 - mortality_recomputed['periodized_calibrated_probability']
    mortality_recomputed['final_state'] = death_state

    periodized_calibrated_transitions = (pd.concat([
        # mortality.reset_index()[
        #     ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
        #     ],
        mortality_recomputed[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        other_transitions[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        ]))

    try:
        assert_probabilities(
            dataframe = periodized_calibrated_transitions,
            by = ['period', 'sex', 'age', 'initial_state'],
            probability = 'periodized_calibrated_probability',
            cut_off = delta,
            )
    except AssertionError as e:
        ValueError("Problem with probabilities for survival_gain_cast = {}:\n {}".format(
            survival_gain_cast, e))

    return (periodized_calibrated_transitions
        .rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )


# def add_projection_corrections(mortality_calibrated_target, mu = None, variant = None, period = None, periods = None):
#     """
#     Take a mortality calibrated target on a specific period = period and project if on periods
#     (subsequent periods if None)
#     """
#     death_state = 4
#     assert variant is None or variant in [0, 1]
#     if variant is not None:
#         assert mu is not None
#     delta = 1e-7
#     regularize(
#         transition_matrix_dataframe = mortality_calibrated_target,
#         by = ['period', 'sex', 'age', 'initial_state'],
#         probability = 'calibrated_probability',
#         delta = delta)

#     if period is None:
#         period = 2010
#         log.info('period is not set: using 2010')
#     projected_mortality = get_insee_projected_mortality()
#     initial_mortality = (projected_mortality
#         .query('year == @period')
#         .rename(columns = {'mortality': 'initial_mortality'})
#         .reset_index()
#         .drop('year', axis = 1)
#         )
#     if periods is None:
#         periods_query = 'year >= 2010'
#     else:
#         periods_query = 'year in {}'.format(periods)
#         log.debug('periods_query: {}'.format(periods_query))

#     correction_coefficient = (projected_mortality.reset_index()
#         .query(periods_query)
#         .merge(initial_mortality)
#         .eval('correction_coefficient = mortality / initial_mortality', inplace = False)
#         .rename(columns = dict(year = 'period'))
#         .drop(['mortality', 'initial_mortality'], axis = 1)
#         )

#     assert not (correction_coefficient['correction_coefficient'].isnull().any())

#     uncalibrated_probabilities = (mortality_calibrated_target.reset_index()[
#         ['sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
#         ]
#         .merge(correction_coefficient[['period', 'sex', 'age', 'correction_coefficient']])
#         .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
#         .sort_index()
#         )

#     assert not (uncalibrated_probabilities['calibrated_probability'].isnull().any()), \
#         "There are {} NaN(s) in uncalibrated_probabilities".format(
#             uncalibrated_probabilities['calibrated_probability'].isnull().sum())

#     assert (
#         uncalibrated_probabilities.query('final_state == @death_state')['calibrated_probability'] < 1
#         ).all(), "There are {} 1's in calibrated_probability".format(
#             (uncalibrated_probabilities['calibrated_probability'] < 1).sum(),
#             )

#     assert_probabilities(
#         dataframe = uncalibrated_probabilities,
#         by = ['period', 'sex', 'age', 'initial_state'],
#         probability = 'calibrated_probability',
#         )

#     mortality = (uncalibrated_probabilities
#         .query('final_state == 5')
#         ).copy()

#     mortality['periodized_calibrated_probability'] = np.minimum(  # use minimum to avoid over corrections !
#         mortality.calibrated_probability * mortality.correction_coefficient, 1 - delta)

#     assert (
#         (mortality['periodized_calibrated_probability'] < 1)
#         ).all(), "There are {} 1's in periodized_calibrated_probability".format(
#             (mortality['periodized_calibrated_probability'] < 1).sum(),
#             )

#     assert not (mortality['periodized_calibrated_probability'].isnull().any()), \
#         "There are calibrated_probability NaNs in mortality"

#     if mu is None:
#         assert (mortality.calibrated_probability < 1).all()
#         mortality.eval(
#             'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
#             inplace = True,
#             )
#         assert not mortality.cale_other_transitions.isnull().any(), "Some calibration coeffecients are NaNs"
#         assert (mortality.cale_other_transitions > 0).all(), "Some calibration coeffecients are negative"

#         cale_other_transitions = mortality.reset_index()[
#             ['period', 'sex', 'age', 'initial_state', 'cale_other_transitions']
#             ].copy()
#         other_transitions = (uncalibrated_probabilities
#             .reset_index()
#             .query('final_state != @death_state')
#             .merge(cale_other_transitions)
#             .eval(
#                 'periodized_calibrated_probability = calibrated_probability * cale_other_transitions',
#                 inplace = False,
#                 )
#             )
#         # Ensures that the mortality is the projected one assuming no variation in elderly disability distribution
#         assert not (other_transitions['periodized_calibrated_probability'].isnull().any()), \
#             "There are {} NaN(s) in other_transitions".format(
#                 other_transitions['periodized_calibrated_probability'].isnull().sum())
#         assert (
#             (other_transitions['periodized_calibrated_probability'] >= 0) &
#             (other_transitions['periodized_calibrated_probability'] <= 1)
#             ).all(), "Erroneous periodized_calibrated_probability"
#     else:
#         assert variant is not None
#         if variant == 0:
#             other_transitions = initial_vs_others(
#                 mortality = mortality, mu = mu, uncalibrated_probabilities = uncalibrated_probabilities)
#         elif variant == 1:
#             other_transitions = autonomy_vs_disability(
#                 mortality = mortality, mu = mu, uncalibrated_probabilities = uncalibrated_probabilities)
#         else:
#             raise NotImplementedError

#     mortality_recomputed = (other_transitions
#         .query('final_state != @death_state')
#         .groupby(
#         ['period', 'sex', 'age', 'initial_state']
#             )['periodized_calibrated_probability'].sum()
#         )
#     mortality_recomputed['periodized_calibrated_probability'] = 1 - mortality_recomputed['periodized_calibrated_probability']
#     mortality_recomputed['final_state'] = death_state

#     periodized_calibrated_transitions = (pd.concat([
#         # mortality.reset_index()[
#         #     ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
#         #     ],
#         mortality_recomputed[
#             ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
#             ],
#         other_transitions[
#             ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
#             ],
#         ]))

#     assert_probabilities(
#         dataframe = periodized_calibrated_transitions,
#         by = ['period', 'sex', 'age', 'initial_state'],
#         probability = 'periodized_calibrated_probability',
#         cut_off = delta,
#         )
#     return (periodized_calibrated_transitions
#         .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
#         .sort_index()
#         )


def initial_vs_others(period = None, mortality = None, mu = None, uncalibrated_probabilities = None):
    """
    Gain in survival probability feeds by a proportion of mu the transition probability towards the initial_state
    and 1 - mu transition probability towards the other states
    """
    assert period is not None
    death_state = 4
    mortality = (mortality
        .query('period == @period')
        .eval(
            'delta = - (periodized_calibrated_probability - calibrated_probability)',
            inplace = False,
            )
        )
    # mortality.delta = np.maximum(mortality.delta, 0)
    assert (mortality.delta >= 0).all(), "No mortality gain for {}".format(mortality.loc[mortality.delta < 0])

    other_transitions = pd.DataFrame()
    for initial_state, final_states in final_states_by_initial_state.iteritems():
        if death_state not in final_states:
            continue
        #

        delta = mortality.reset_index()[
            ['period', 'sex', 'age', 'initial_statrecompe', 'delta']
            ].copy()

        to_initial_transitions_probability = (uncalibrated_probabilities
            .reset_index()
            .query('(initial_state == @initial_state) and (final_state == @initial_state) and (final_state != @death_state)')
            .merge(delta)[
                ['period', 'sex', 'age', 'initial_state', 'calibrated_probability', 'delta']
                ]
            .copy()
            )

        non_initial_transitions_aggregate_probability = (uncalibrated_probabilities
            .reset_index()
            .query('(initial_state == @initial_state) and (final_state != @initial_state) and (final_state != @death_state)')
            .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
            .reset_index()
            .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
            )

        adjusted_mu = (to_initial_transitions_probability
            .merge(non_initial_transitions_aggregate_probability)
            .eval('mu = @mu', inplace = False)
            )
        adjusted_mu['mu'] = np.where(
            adjusted_mu.mu <= adjusted_mu.calibrated_probability / adjusted_mu.delta,
            adjusted_mu.mu,
            adjusted_mu.calibrated_probability / adjusted_mu.delta,
            )
        adjusted_mu['mu'] = np.where(
            adjusted_mu.mu >= 1 - adjusted_mu.aggregate_calibrated_probability / adjusted_mu.delta,
            adjusted_mu.mu,
            1 - adjusted_mu.aggregate_calibrated_probability / adjusted_mu.delta,
            )

        adjusted_mu = adjusted_mu.reset_index()[
            ['period', 'sex', 'age', 'initial_state', 'mu', 'delta']
            ].copy()

        to_initial_transitions = (uncalibrated_probabilities
            .reset_index()
            .query('(initial_state == @initial_state) and (final_state == @initial_state) and (final_state != @death_state)')
            .merge(adjusted_mu)
            .eval(
                'periodized_calibrated_probability = calibrated_probability + mu * delta',
                inplace = False,
                )
            )
        to_non_initial_transitions = (uncalibrated_probabilities
            .reset_index()
            .query('(initial_state == @initial_state) and (final_state != @initial_state) and (final_state != @death_state)')
            .merge(non_initial_transitions_aggregate_probability.reset_index())
            .merge(adjusted_mu)
            .eval(
                'periodized_calibrated_probability = calibrated_probability * (1 + (1 - mu) * delta / aggregate_calibrated_probability)',
                inplace = False,
                )
            )

        assert (to_initial_transitions['periodized_calibrated_probability'] >= 0).all()
        assert (to_initial_transitions['periodized_calibrated_probability'] <= 1).all()
        assert (to_non_initial_transitions['periodized_calibrated_probability'] <= 1).all()
        assert (to_non_initial_transitions['periodized_calibrated_probability'] >= 0).all(), \
            to_non_initial_transitions.loc[to_non_initial_transitions['periodized_calibrated_probability'] < 0]

        other_transitions = pd.concat([
            other_transitions,
            to_initial_transitions[
                ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
                ],
            to_non_initial_transitions[
                ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
                ],
            ])

        assert not other_transitions[['period', 'sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
            'Duplicated transitions for initial_state = {}: {}'.format(
                initial_state,
                other_transitions.loc[
                    other_transitions[['period', 'sex', 'age', 'initial_state', 'final_state']].duplicated()
                    ]
                )

    return other_transitions


def autonomy_vs_disability(mortality = None, mu = None, uncalibrated_probabilities = None):
    """
    Autonomy and disability states are treated differently
    Gain in survival probability feeds by a proportion of mu the autonomy initial_state and
    1 - mu the transition to the disability states.
    Other transition share proportionnally the survival probability gain
    """
    assert mu is not None
    death_state = 4
    # Starting from autonomy
    # Staying autonomous

    mortality.eval(
        'survival_gain = - (periodized_calibrated_probability - calibrated_probability)',  # It is postive gain
        inplace = True,
        )

    mortality.survival_gain = np.maximum(mortality.survival_gain, 0)

    stay_autonomous = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state == 0)')
        )

    become_disabled = (uncalibrated_probabilities
        .reset_index()
        # .query('(initial_state == 0) & (final_state != 0) & (final_state != @death_state)')
        .query('(initial_state == 0) & (final_state == 1)')
        .rename(columns = dict(calibrated_probability = 'become_disabled_calibrated_probability'))
        ) # .merge(disabling_transitions_aggregate_probability)

    survival_gain = mortality.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'survival_gain']
        ].copy()

    adjusted_mu = (stay_autonomous[['period', 'sex', 'age', 'initial_state', 'calibrated_probability']]
        .merge(become_disabled[['period', 'sex', 'age', 'initial_state', 'become_disabled_calibrated_probability']])
        .merge(survival_gain)
        .eval('mu = @mu', inplace = False)
        )

    adjusted_mu['mu'] = np.where(
        adjusted_mu.mu <= (adjusted_mu.calibrated_probability / adjusted_mu.survival_gain),
        adjusted_mu.mu,
        adjusted_mu.calibrated_probability / adjusted_mu.survival_gain,
        )
    adjusted_mu['mu'] = np.where(
        adjusted_mu.mu >= (1 - adjusted_mu.become_disabled_calibrated_probability / adjusted_mu.survival_gain),
        adjusted_mu.mu,
        1 - adjusted_mu.become_disabled_calibrated_probability / adjusted_mu.survival_gain,
        )

    adjusted_mu = adjusted_mu.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'mu', 'survival_gain']
        ].copy()

    stay_autonomous = (stay_autonomous
        .merge(adjusted_mu)
        .eval(
            'periodized_calibrated_probability = mu * survival_gain + calibrated_probability',
            inplace = False,
            )
        )
    assert (stay_autonomous.periodized_calibrated_probability <= 1).all()
    assert (stay_autonomous.periodized_calibrated_probability >= 0).all(), \
        stay_autonomous.loc[stay_autonomous.periodized_calibrated_probability < 0]

    # # From autonomy to disablement
    # disabling_transitions_aggregate_probability = (uncalibrated_probabilities
    #     .reset_index()
    #     .query('(initial_state == 0) & (final_state != 0) & (final_state != @death_state)')
    #     .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
    #     .reset_index()
    #     .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
    #     )

    become_disabled = (become_disabled
        .merge(adjusted_mu)
        .eval(
            'periodized_calibrated_probability =  become_disabled_calibrated_probability * (1 + (1 - mu) * survival_gain / become_disabled_calibrated_probability)',
            inplace = False,
            )
        .rename(columns = dict(become_disabled_calibrated_probability= 'calibrated_probability '))
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
        .query('(initial_state != 0) & (final_state != @death_state)')
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


def assert_probabilities(dataframe = None, by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability', cut_off = 1e-8):
    assert dataframe is not None
    assert not (dataframe[probability] < 0).any(), dataframe.loc[dataframe[probability] < 0]
    assert not (dataframe[probability] > 1).any(), dataframe.loc[dataframe[probability] > 1]
    diff = (
        dataframe.reset_index().groupby(by)[probability].sum() - 1)
    diff.name = 'error'
    assert (diff.abs().max() < cut_off).all(), "error is too big: {} > {}. Example: {}".format(
        diff.abs().max(), cut_off, (dataframe
            .reset_index()
            .set_index(by)
            .loc[diff.abs().argmax(), ['final_state', probability]]
            .reset_index()
            .set_index(by + ['final_state'])
            )
        )


def build_mortality_calibrated_target(transitions = None, period = None, dependance_initialisation = None, scale = 4,
        age_min = None):
    """
    Compute the calibrated mortality by sex, age and disability state (initial_state) for a given period
    using data on the disability states distribution in the population at that period
    if dependance_initialisation = None
    TODO should be merged with build_mortality_calibrated_target_from_transitions
    """
    assert age_min is not None
    assert scale in [4, 5]
    if scale == 5:
        death_state = 5
    elif scale == 4:
        death_state = 4

    assert (transitions is not None) and (period is not None)
    calibrated_transitions = _get_calibrated_transitions(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        )

    null_fill_by_year = calibrated_transitions.reset_index().query('age == @age_min').copy()
    null_fill_by_year['calibrated_probability'] = 0

    # Less than ageè_min years old -> no correction
    pre_age_min_null_fill = pd.concat([
        null_fill_by_year.assign(age = i).copy()
        for i in range(0, age_min)
        ]).reset_index(drop = True)

    pre_age_min_null_fill.loc[
        pre_age_min_null_fill.initial_state == pre_age_min_null_fill.final_state,
        'calibrated_probability'
        ] = 1

    assert_probabilities(
        dataframe = pre_age_min_null_fill,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    # More than age_min years old
    age_max = calibrated_transitions.index.get_level_values('age').max()
    if age_max < 120:
        elder_null_fill = pd.concat([
            null_fill_by_year.assign(age = i).copy()
            for i in range(age_max + 1, 121)
            ]).reset_index(drop = True)

        elder_null_fill.loc[
            (elder_null_fill.age > age_max) & (elder_null_fill.final_state == death_state),
            'calibrated_probability'
            ] = 1
        assert_probabilities(
            dataframe = elder_null_fill,
            by = ['sex', 'age', 'initial_state'],
            probability = 'calibrated_probability',
            )
        age_full = pd.concat([
            pre_age_min_null_fill,
            calibrated_transitions.reset_index(),
            elder_null_fill
            ]).reset_index(drop = True)
    else:
        age_full = pd.concat([
            pre_age_min_null_fill,
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


def build_mortality_calibrated_target_from_formula(formula = None, period = None, age_min = None):
    assert period is not None
    assert formula is not None
    transitions = get_transitions_from_formula(formula = formula)
    return build_mortality_calibrated_target_from_transitions(
        transitions = transitions,
        period = period,
        age = age_min,
        )


def build_mortality_calibrated_target_from_transitions(transitions = None, period = None, dependance_initialisation = None,
       age_min = None):
    assert age_min is not None
    assert period is not None
    assert transitions is not None
    mortality_calibrated_target = build_mortality_calibrated_target(
        transitions = transitions,
        period = period,
        dependance_initialisation = dependance_initialisation,
        age_min = age_min,
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
            filename = os.path.join(input_dir, 'dependance_transition_merged_0_1_{}.csv'.format(sex))
        else:
            filename = os.path.join('{}_{}.csv'.format(filename_prefix, sex))

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


def get_mortality_after_imputation(mortality_table = None, dependance_initialisation = None, age_min = 50):
    """
    Compute total mortality from mortality by dependance initial state given dependance_initialisation_male/female.csv
    present in til/input_dir or from dependance_initialisation data_frame if not None
    """

    assert mortality_table is not None
    if dependance_initialisation is not None:
        data = dependance_initialisation.rename(columns = {'population': 'total'})
    else:
        raise NotImplementedError
    mortality_after_imputation = (data
        .merge(
            mortality_table.reset_index()[['sex', 'age', 'initial_state', 'mortality']],
            on = ['sex', 'age', 'initial_state'],
            how = 'inner',
            )
        .groupby(['sex', 'age'])[['total', 'mortality']].apply(lambda x: (
            (x.total * x.mortality).sum() / (x.total.sum() + (x.total.sum() == 0))  # Use last term to avoid 0 / 0
            ))
        )

    mortality_after_imputation.name = 'mortality_after_imputation'

    if (mortality_after_imputation == 0).any():  # Deal with age > 100 with nobody in the population
        mortality_after_imputation = mortality_after_imputation.reset_index()
        mortality_after_imputation.loc[
            (mortality_after_imputation.age >= 100) & (mortality_after_imputation.mortality_after_imputation == 0),
            'mortality_after_imputation'] = .6
        mortality_after_imputation.set_index(['sex', 'age'])

    return mortality_after_imputation


def _compute_calibration_coefficient(age_min = 50, period = None, transitions = None, dependance_initialisation = None):
    """
    Calibrate mortality using the distribution of the disability states within population at a specific year
    for the given transition matrix and distribution of intiial_states
    Assuming the transition occur on a two-year period.
    """
    assert period is not None, "Mortality profile period is not set"
    assert transitions is not None
    # From 2yr mortality to 1yr mortality by age, sex and intial_state
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions)
    # From 1yr mrotality by age sex (use dependance_initialisation to sum over initial_state)
    mortality_after_imputation = (
        get_mortality_after_imputation(
            mortality_table = predicted_mortality_table,
            dependance_initialisation = dependance_initialisation,
            )
        .reset_index()
        .rename(columns = {'mortality_after_imputation': 'avg_mortality'})
        )

    assert (mortality_after_imputation.avg_mortality > 0).all(), \
        mortality_after_imputation.loc[~(mortality_after_imputation.avg_mortality > 0)]

    projected_mortality = (get_insee_projected_mortality()  # brings in variable mortality
        .query('year == @period')
        .rename(columns = {'year': 'period'})
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
    death_state = 4
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
        .query('final_state == @death_state')
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
        .query('final_state != @death_state')
        .merge(
            cale_other_transitions,
            on = ['sex', 'age', 'initial_state'],
            )
        .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
        )
    assert other_transitions.calibrated_probability.notnull().all(), \
        other_transitions.loc[other_transitions.calibrated_probability.isnull()]

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
                pd.read_excel(life_table_path, sheet_name = 'france-{}'.format(sex))[['Year', 'Age', 'qx']]
                .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
                .replace(dict(age = {'110+': '110'}))
                )
            sex_historical_mortality['age'] = sex_historical_mortality['age'].astype('int')
            sex_historical_mortality['sex'] = sex
            sex_historical_mortalities.append(sex_historical_mortality)

        historical_mortality = pd.concat(sex_historical_mortalities)
        historical_mortality.to_hdf(mortality_store_path, key = 'historical_mortality')
    return historical_mortality


def get_predicted_mortality_table(transitions = None, save = True, probability_name = 'probability'):
    death_state = 4
    assert transitions is not None
    assert probability_name in transitions.columns, "{} not present in transitions colmns: {}".format(
        probability_name,
        transitions.columns
        )
    mortality_table = (transitions
        .query('final_state == @death_state')
        .copy()
        .assign(mortality = lambda x: (1 - np.sqrt(1 - x[probability_name])))
        )
    if save:
        mortality_table.to_csv('predicted_mortality_table.csv')

    mortality_table.loc[
        ~(mortality_table.mortality > 0),
        'mortality'
        ] = 1e-12  # Avoid null mortality
    assert (mortality_table.mortality > 0).all(), mortality_table.loc[~(mortality_table.mortality > 0)]

    return mortality_table


def get_insee_projected_mortality():
    '''
    Get mortality data from INSEE projections
    '''
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
        for sex, sheet_name in sheet_name_by_sex.iteritems()
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


def regularize(transition_matrix_dataframe = None, by = None, probability = None, delta = None):
    assert transition_matrix_dataframe is not None
    assert by is not None
    assert probability is not None
    assert delta is not None
    death_state = 4
    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    mortality_transitions = transition_matrix_dataframe.query('final_state == @death_state').copy()

    # by_without_initial_state = [by_value for by_value in by if by_value != 'initial_state']
    problematic_indices = (mortality_transitions[probability]
        .loc[mortality_transitions[probability] >= (1 - delta)]
        .reset_index()
        .drop(['final_state', probability], axis = 1)
        )
    count = (problematic_indices
        .merge(
            transition_matrix_dataframe.reset_index()[by + ['final_state']])
        .query('final_state != @death_state')
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
        (corrected_transition_matrix.final_state == death_state),
        probability
        ] = (1 - delta)
    corrected_transition_matrix.loc[
        corrected_transition_matrix.final_state != death_state,
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
        transition_matrix_dataframe.query('final_state == @death_state')[probability] < 1
        ).all(), "There are {} 1's in transition_matrix_dataframe.{}".format(
            (transition_matrix_dataframe.query('final_state == @death_state')[probability] < 1).sum(),
            probability,
            )

    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    return transition_matrix_dataframe


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)

    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1)**2) + I(((age - 80) * 0.1)**3)'
    uncalibrated_transitions = get_transitions_from_formula(formula = formula)
    print uncalibrated_transitions
    BIM

    config = Config()
    input_dir = config.get('til', 'input_dir')

    def plot_dependance_niveau_by_age(population, period, sexe = None, area = False, pct = True):
        assert 'period' in population.columns, 'period is not present in population columns: {}'.format(population.columns)
        assert period in population.period.unique()
        data = population.query('period == @period').copy()
        if 'initial_state' in data.columns:
            data.rename(columns = {'initial_state': 'dependance_niveau'}, inplace = True)
        pivot_table = (data
            .groupby(['period', 'age', 'dependance_niveau'])['population'].sum().reset_index()
            .pivot('age', 'dependance_niveau', 'population')
            )
        print pivot_table

        if pct:
            pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)

        # Remove all 0 columns
        pivot_table = (pivot_table
            .replace(0, np.nan)
            .dropna(how = 'all', axis = 1)
            .replace(np.nan, 0)
            )
        if area:
            pivot_table.plot.area(stacked = True, color = colors)
        else:
            pivot_table.plot.line(stacked = True, color = colors)


    population_1.groupby(['period', 'initial_state'])['population'].sum().unstack().plot()
    (population_1.set_index(['period', 'initial_state', 'age', 'sex']) - population_0.set_index(['period', 'initial_state', 'age', 'sex'])).groupby(['period', 'initial_state'])['population'].sum().unstack().plot()


    population_1.groupby(['period', 'initial_state'])['population'].sum().unstack().plot()


    pivot_table = population_1.groupby(['period', 'initial_state'])['population'].sum().unstack()

    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
    pct_pivot_table.plot.line()

    pivot_table_1 = population_1.groupby(['period', 'initial_state'])['population'].sum().unstack()
    pivot_table_0 = population_0.groupby(['period', 'initial_state'])['population'].sum().unstack()

    pct_pivot_table_1 = pivot_table_1.divide(pivot_table_1.sum(axis=1), axis=0)
    pct_pivot_table_0 = pivot_table_0.divide(pivot_table_0.sum(axis=1), axis=0)

    (pct_pivot_table_1 - pct_pivot_table_0).plot.line()
