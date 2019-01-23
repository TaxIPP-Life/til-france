# -*- coding: utf-8 -*-


import logging
import numpy as np
import os
import pandas as pd


from til_france.model.options.dependance_RT.life_expectancy.share.insee import get_insee_projected_mortality
from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import final_states_by_initial_state

log = logging.getLogger(__name__)


def build_mortality_calibrated_target_from_transitions(transitions = None, period = None, dependance_initialisation = None,
       age_min = None, one_year_approximation = None, survival_gain_cast = None, mu = None, age_max_cale = None, uncalibrated_transitions = None):
    """
    Compute the calibrated mortality by sex, age and disability state (initial_state) for a given period
    using data on the disability states distribution in the population at that period
    if dependance_initialisation = None
    TODO should be merged with build_mortality_calibrated_target_from_transitions
    """
    assert age_min is not None
    assert period is not None
    assert transitions is not None
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

    death_state = 4  # ie scale = 4

    assert (transitions is not None) and (period is not None)
    calibrated_transitions = _get_calibrated_transitions(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        one_year_approximation = one_year_approximation,
        survival_gain_cast = survival_gain_cast,
        mu = mu,
        age_max_cale = age_max_cale,
        uncalibrated_transitions = uncalibrated_transitions
        )

    null_fill_by_year = calibrated_transitions.reset_index().query('age == @age_min').copy()
    null_fill_by_year['calibrated_probability'] = 0

    # Less than age_min years old -> no correction
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


def get_mortality_after_imputation(mortality_table = None, dependance_initialisation = None, age_min = 50):
    """
        Compute total mortality from mortality by dependance initial state

        These states are given by the files dependance_initialisation_male/female.csv
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
        log.info("missing dans mortality_after_imputation")
        mortality_after_imputation = mortality_after_imputation.reset_index()
        mortality_after_imputation.loc[
            (mortality_after_imputation.age >= 100) & (mortality_after_imputation.mortality_after_imputation == 0),
            'mortality_after_imputation'] = .84  # Mortalité à deux ans pour une mortalite à 1 an de 0.6, ie .84 = .6 + (1 - .6) * .6
        mortality_after_imputation.set_index(['sex', 'age'])

    return mortality_after_imputation


def get_predicted_mortality_table(transitions = None, save = False, probability_name = 'probability', one_year_approximation = None):
    """
        Get mortality from simulation data
    """
    death_state = 4
    assert transitions is not None
    assert one_year_approximation is not None
    assert probability_name in transitions.columns, "{} not present in transitions columns: {}".format(
        probability_name,
        transitions.columns
        )

    if not one_year_approximation:
        # Transitions vers deces avec un pas de 2 ans
        mortality_table = (transitions
            .query('final_state == @death_state')
            .copy()
            .assign(mortality = lambda x: x[probability_name])
            )
    else:
        # one_year_approximation is true : transitions vers deces avec un pas de 1 an
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


def _compute_calibration_coefficient(age_min = 50, period = None, transitions = None, dependance_initialisation = None,
        one_year_approximation = None):
    """
        Calibrate mortality using the distribution of the disability states within population at a specific year
        for the given transition matrix and distribution of intiial_states

        Assuming the transition occur on a two-year period.

        From 2 years mortality to 1 year mortality by age, sex and intial_state if one_year_approximation is True
        2 years mortality by age, sex and intial_state if one_year_approximation is False
    """
    assert period is not None, "Mortality profile period is not set"
    assert transitions is not None
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions, one_year_approximation = one_year_approximation)
    # From 1 year mortality by age sex (use dependance_initialisation to sum over initial_state)
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

    mortality_insee =(get_insee_projected_mortality()
        .query('year == @period')
        .rename(columns = {'year': 'period'})
        )

    assert mortality_insee.mortality_insee_next_period is not None

    model_to_target = (mortality_after_imputation
        .merge(
            mortality_insee.reset_index(),
            on = ['sex', 'age'],
            )
        )

    if not one_year_approximation: # transformations à 2 ans pour l'instant
        model_to_target = (model_to_target
            .eval('avg_mortality_2_year = avg_mortality', inplace = False)
            .eval('cale_mortality_2_year = mortalite_2_year_insee / avg_mortality_2_year', inplace = False)
            )

    return model_to_target


def _get_calibrated_transitions(period = None, transitions = None, dependance_initialisation = None, one_year_approximation = None,
        survival_gain_cast = None, mu = None, age_max_cale = None, uncalibrated_transitions = None):
    """
        Calibrate transitions to match mortality from a specified period
    """
    death_state = 4
    assert (period is not None) and (transitions is not None)
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

    # Add calibration_coeffcients for mortality
    calibration = _compute_calibration_coefficient(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        one_year_approximation = one_year_approximation,
        )

    # Selectionne calibration jusque age_max
    calibration = calibration.loc[calibration['age'] < age_max_cale]

    assert not calibration.reset_index()[['sex', 'age']].duplicated().any(), \
        calibration.reset_index().loc[calibration.reset_index()[['sex', 'age']].duplicated()]

    # Calibrate mortality
    assert not transitions.reset_index()[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
        transitions.reset_index().loc[transitions.reset_index()[
            ['sex', 'age', 'initial_state', 'final_state']].duplicated()]

    mortality = (transitions
        .reset_index()
        .query('final_state == @death_state')
        .query('age < @age_max_cale') #a remplacer par age_max_cale
        .merge(
            calibration[['sex', 'age', 'cale_mortality_2_year']].copy(),
            on = ['sex', 'age'],
            )
        )
    mortality['calibrated_probability'] = np.minimum(
        mortality.probability * mortality.cale_mortality_2_year, 1)  # Avoid over corrections !

    # Création des beta pour le scénario homogeneous
    if survival_gain_cast == 'homogeneous':
        log.debug("Using homogeneous scenario")
        mortality.eval(
            'cale_other_transitions = (1 - calibrated_probability) / (1 - probability)',
            inplace = True,
            )
        assert not mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), mortality.loc[mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated()]

        # Calibrate other transitions
        cale_other_transitions = (mortality[['sex', 'age', 'initial_state', 'cale_other_transitions']]
            .copy()
            .reset_index()
            .set_index(['sex', 'age', 'initial_state'])
        )
        other_transitions = (transitions
            .reset_index()
            .set_index(['sex', 'age', 'initial_state'])
            .query('final_state != @death_state')
            .merge(
                cale_other_transitions,
                on = ['sex', 'age', 'initial_state'],
                )
            .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
            )

        log.debug("Utilise les cales pour calibrated_probability")
        assert other_transitions.calibrated_probability.notnull().all(), \
            other_transitions.loc[other_transitions.calibrated_probability.isnull()]

    # Création des beta pour le scénario initial_vs_others
    elif survival_gain_cast == "initial_vs_others":
        log.debug("Using initial_vs_others scenario")
        assert mu is not None
        mortality = (mortality
            .rename(columns = {'calibrated_probability': 'periodized_calibrated_probability'})
            )
        # Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
        other_transitions = initial_vs_others(
            # period = period,
            mortality = mortality.rename(
                columns = {'probability': 'calibrated_probability'}
                ),
            mu = mu,
            uncalibrated_probabilities = transitions.rename(columns = {'probability': 'calibrated_probability'})
            )

        other_transitions = (other_transitions
            .set_index(['sex', 'age'])
            .sort_index()
            .fillna(method = 'ffill') #Remplit avec la valeur de l'age precedent pour eviter des missings aux ages eleves
            .reset_index()
            .set_index(['sex', 'age', 'initial_state', 'final_state'])
            )

        # Keep columns we want (en plus de l'index)
        other_transitions = other_transitions[['periodized_calibrated_probability']]

        # Calibrate other transitions
        other_transitions = (transitions
            .reset_index()
            .set_index(['sex', 'age', 'initial_state', 'final_state'])
            .query('final_state != @death_state')
            .merge(
                other_transitions,
                on = ['sex', 'age', 'initial_state', 'final_state'],
                )
            .rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})
            .reset_index()
            .set_index(['sex', 'age'])
            .sort_index()
            .fillna(method = 'ffill') # Remplit avec la valeur de l'âge précédent pour éviter des missings aux âges élevés
            .reset_index()
            .set_index(['sex', 'age', 'initial_state', 'final_state'])
            )

    # Création des beta pour le scénario autonomy_vs_disability
    elif survival_gain_cast == 'autonomy_vs_disability':
        log.debug("Using autonomy_vs_disability scenario")
        assert mu is not None
        mortality = (mortality
            .rename(columns = {'calibrated_probability': 'periodized_calibrated_probability'})
            )
        other_transitions = autonomy_vs_disability(
            mortality = mortality.rename(columns = {'probability': 'calibrated_probability'}),
            mu = mu,
            uncalibrated_probabilities = transitions.rename(columns = {'probability': 'calibrated_probability'})
            )

        # Keep columns we want (en plus de l'index)
        other_transitions = other_transitions[['periodized_calibrated_probability']]
        other_transitions = other_transitions.rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})

    else:
            raise NotImplementedError

    calibrated_transitions = pd.concat(
        [
            mortality.reset_index().set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index(),
            other_transitions.reset_index().set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index()
            ],
        sort = True
        ).sort_index()

    # Remplace les missings de calibrated proba par periodized calibrated proba si final_state=4 pour avoir une variable unique
    if (calibrated_transitions['calibrated_probability']).isnull().any():
        calibrated_transitions['calibrated_probability'] = calibrated_transitions['calibrated_probability'].fillna(value=calibrated_transitions['periodized_calibrated_probability'])

    # Vérification
    assert_probabilities(
        calibrated_transitions, by = ['sex', 'age', 'initial_state'], probability = 'calibrated_probability')

    # Impute les transitions aux ages eleves
    calibrated_transitions = impute_high_ages(
            data_to_complete = calibrated_transitions,
            uncalibrated_transitions = uncalibrated_transitions,
            period = period,
            age_max_cale = age_max_cale
        )
    return calibrated_transitions['calibrated_probability']


# Scénarios

def initial_vs_others(mortality = None, mu = None, uncalibrated_probabilities = None):
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
    other_transitions = pd.DataFrame()
    for initial_state, final_states in final_states_by_initial_state.items():
        if 4 not in final_states:
            continue
        else:
            delta_initial_transitions = mortality.reset_index()[
                ['period', 'sex', 'age', 'initial_state', 'delta_initial']
                ].copy()
            to_initial_transitions = (uncalibrated_probabilities
                .reset_index()
                .query('(initial_state == @initial_state) and (final_state == @initial_state) and (final_state != 4)')
                .merge(delta_initial_transitions)
                )

            to_initial_transitions['periodized_calibrated_probability'] = np.maximum(
                to_initial_transitions.calibrated_probability + to_initial_transitions.delta_initial, 0)

            delta_non_initial_transitions = mortality.reset_index()[
                ['period', 'sex', 'age', 'initial_state', 'delta_non_initial']
                ].copy()

            non_initial_transitions_aggregate_probability = (uncalibrated_probabilities
                .reset_index()
                .query('(initial_state == @initial_state) and (final_state != @initial_state) and (final_state != 4)')
                .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
                .reset_index()
                .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
                )

            to_non_initial_transitions = (uncalibrated_probabilities
                .reset_index()
                .query('(initial_state == @initial_state) and (final_state != @initial_state) and (final_state != 4)')
                .merge(non_initial_transitions_aggregate_probability.reset_index())
                .merge(delta_non_initial_transitions)
                .eval(
                    'periodized_calibrated_probability = calibrated_probability * (1 + delta_non_initial / aggregate_calibrated_probability)',
                    inplace = False,
                    )
                )

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

        :param DataFrame mortality: moratlity table
        :param mu mortality: Share of survival gain casted to the autonomy state
        :param DataFrame uncalibrated_probabilities: probability table to be updated
    """
    # Starting from autonomy
    # Staying autonomous
    assert mu is not None
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
        .query('(initial_state == 0) & (final_state != 0) & (final_state != 4)')
        .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
        .reset_index()
        .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
        )
    become_disabled = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state != 0) & (final_state != 4)')
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
        .query('(initial_state != 0) & (final_state != 4)')
        .merge(cale_other_transitions)
        .eval(
            'periodized_calibrated_probability = calibrated_probability * cale_other_transitions',
            inplace = False,
            )
        )

    all_transitions = pd.concat(
        [stay_autonomous, become_disabled, other_transitions,],
        sort = True,
        )

    all_transitions = (all_transitions
        .reset_index()
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        )

    return all_transitions


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


def impute_high_ages(data_to_complete = None, uncalibrated_transitions = None, period = None, age_max_cale = None):
    """
        Impute probability transition for high ages by forward filling values using a threshold age

        :param DataFrame data_to_complete: data to be complted by imputation
        :param DataFrame uncalibratd_transitions: disability transitions table
        :param int period: the period of the simulation
        :param int age_max_cale: minimal age at which the disability level transitions are freezed
    """

    assert age_max_cale is not None
    assert uncalibrated_transitions is not None
    transitions_data = (uncalibrated_transitions
        .reset_index()
        .query('age >= @age_max_cale')
        .assign(calibrated_probability = np.nan)
        .set_index(['sex', 'age', 'initial_state', 'final_state'])
        )

    complete_data = pd.concat([data_to_complete, transitions_data], sort = True)
    complete_data = (complete_data
        .reset_index()
        .set_index(['sex', 'initial_state', 'final_state', 'age'])
        .sort_index()
        .fillna(method = 'ffill')  # forward fill to avoid missings valeus at high ages
        .reset_index()
        .set_index(['sex', 'age', 'initial_state', 'final_state'])
        )

    return complete_data
