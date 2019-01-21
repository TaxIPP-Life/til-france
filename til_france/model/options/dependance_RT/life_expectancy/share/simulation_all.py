# -*- coding: utf-8 -*-


from __future__ import division


import logging
import numpy as np
import os
import pandas as pd
import slugify


from til_core.config import Config


from til_france.model.options.dependance_RT.life_expectancy.share.prevalence import create_initial_prevalence

from til_france.model.options.dependance_RT.life_expectancy.share.calibration import (
    assert_probabilities,
    build_mortality_calibrated_target_from_transitions,
    get_insee_projected_mortality,
    )

from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    )

from til_france.model.options.dependance_RT.life_expectancy.share.insee import get_insee_projected_population


log = logging.getLogger(__name__)


config = Config()
figures_directory = config.get('dependance', 'figures_directory')
eligible_survival_gain_casts = ['homogeneous', 'initial_vs_others', 'autonomy_vs_disability']


def run(survival_gain_casts = None, mu = None, uncalibrated_transitions = None, vagues = [4, 5, 6], age_min = None,
        prevalence_survey = None, one_year_approximation = None, age_max_cale = None):
    """
        Run

        :param list vagues: share data waves used for transition estimation
        :param int age_min:
        :param DataFrame uncalibrated_transitions:
        :param str prevalence_survey: survey used to compute initial prevalence, should be 'care', 'hsm' or 'hsm_hsi'
        :param int age_max_cale:
        :param list survival_gain_casts: list of eligible survival_gain_casts, should be 'autonomy_vs_disability', 'homogeneous' or 'initial_vs_others',
    """
    assert vagues is not None
    assert age_min is not None
    assert uncalibrated_transitions is not None
    assert prevalence_survey in ['care', 'hsm', 'hsm_hsi']
    assert age_max_cale is not None
    assert set(survival_gain_casts) < set(eligible_survival_gain_casts)

    create_initial_prevalence(smooth = True, prevalence_survey = prevalence_survey, age_min = age_min)
    for survival_gain_cast in survival_gain_casts:
        save_data_and_graph(
            uncalibrated_transitions = uncalibrated_transitions,
            survival_gain_cast = survival_gain_cast,
            vagues = vagues,
            age_min = age_min,
            prevalence_survey = prevalence_survey,
            one_year_approximation = one_year_approximation,
            mu = mu,
            age_max_cale = age_max_cale
            )


def save_data_and_graph(uncalibrated_transitions, mu = None, survival_gain_cast = None, vagues = None, age_min = None, prevalence_survey = None, one_year_approximation = None, age_max_cale = None):
    if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
        assert mu is not None

    log.info("Running with survival_gain_cast = {}".format(survival_gain_cast))
    initial_period = 2010
    initial_population = get_initial_population(age_min = age_min, rescale = True, period = initial_period, prevalence_survey = prevalence_survey)
    initial_population['period'] = initial_period
    population, transitions_by_period = project_disability(
        uncalibrated_transitions = uncalibrated_transitions,
        initial_population = initial_population,
        mu = mu,
        survival_gain_cast = survival_gain_cast,
        age_min = age_min,
        prevalence_survey = prevalence_survey,
        one_year_approximation = one_year_approximation,
        age_max_cale = age_max_cale
        )

    # Save data
    suffix = build_suffix(survival_gain_cast, mu, vagues, prevalence_survey)
    population_path = os.path.join(figures_directory, 'population_{}.csv'.format(suffix))
    log.info("Saving population data to {}".format(population_path))
    population.to_csv(population_path)
    transitions = pd.concat(transitions_by_period.values())
    (transitions
        .reset_index()
        .sort_values(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .to_csv(os.path.join(figures_directory, 'transitions_{}.csv'.format(suffix)))
        )
    pivot_table = population.groupby(['period', 'initial_state'])['population'].sum().unstack()
    pivot_table.to_csv(os.path.join(figures_directory, 'share_proj_{}.csv'.format(
        suffix)))

    # Plot data
    ax = pivot_table.plot.line()
    figure = ax.get_figure()
    figure.savefig(os.path.join(figures_directory, 'share_proj_{}.pdf'.format(
        suffix)), bbox_inches = 'tight')
    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis = 1), axis = 0)
    ax = pct_pivot_table.plot.line()
    figure = ax.get_figure()
    pct_pivot_table.to_csv(os.path.join(figures_directory, 'share_proj_pct_{}.csv'.format(
        suffix)))
    figure.savefig(os.path.join(figures_directory, 'share_proj_pct_{}.pdf'.format(suffix)), bbox_inches = 'tight')


def project_disability(uncalibrated_transitions = None, initial_population = None, initial_period = 2010, mu = None,
        survival_gain_cast = None, age_min = None, prevalence_survey = None, one_year_approximation = None, age_max_cale = None):
    """
        Project disabilyt levels by simulation according to a scenario specifying the survival gain casts

        :param DataFrame uncalibrated_transitions: transition matrix between different disability levels
        :param DataFrame initial_population: population at the initial period
        :param int initial_period: initial period of the simulation
        :param float mu: optional parameter for scenarios, should be in the [0, 1] interval.
        :param str survival_gain_cast: scenario for survival gain casting. Should be 'autonomy_vs_disability', 'homogeneous' or 'initial_vs_others'
        :param int age_min: minimal age for people to be disabled
        :param str prevalence_survey: survey used to compute initial prevalence, should be 'care', 'hsm' or 'hsm_hsi'
        :param one_year_approximation:
        :param int age_max_cale: minimal age at which the disability level transitions are freezed
    """
    assert prevalence_survey is not None
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

    initial_population['period'] = initial_period
    population = initial_population.copy()
    uncalibrated_transitions = (uncalibrated_transitions
        .reset_index()
        .assign(period = initial_period)
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        )
    transitions = build_mortality_calibrated_target_from_transitions(
        transitions = uncalibrated_transitions,
        period = initial_period,
        dependance_initialisation = population,
        age_min = age_min,
        one_year_approximation = one_year_approximation,
        survival_gain_cast = survival_gain_cast,
        mu = mu,
        age_max_cale = age_max_cale,
        uncalibrated_transitions = uncalibrated_transitions
        )
    period = initial_period
    transitions_by_period = dict()

    while period < 2058:
        log.info('Running period {}'.format(period))
        period = population['period'].max()
        if period > initial_period:
            dependance_initialisation = population.query('period == @period').copy()
            # Update the transitions matrix if necessary
            admissible_scenarios = ['homogeneous', 'initial_vs_others', 'autonomy_vs_disability']
            assert survival_gain_cast in admissible_scenarios, "survival_gain_cast should be int the following list:\n  {}".format(
                admissible_scenarios)
            log.info("Calibrate transitions for period = {} usning survival_gain_cast = {}".format(
                period, survival_gain_cast))
            delta = 1e-7
            transitions = regularize(
                transition_matrix_dataframe = transitions.rename(
                    columns = {'calibrated_probability': 'probability'}
                    ),
                by = ['period', 'sex', 'age', 'initial_state'],
                probability = 'probability',
                delta = delta,
                )
            transitions = build_mortality_calibrated_target_from_transitions(
                transitions = transitions,
                period = period,
                dependance_initialisation = dependance_initialisation,
                age_min = age_min,
                one_year_approximation = one_year_approximation,
                survival_gain_cast = survival_gain_cast,
                mu = mu,
                age_max_cale = age_max_cale,
                uncalibrated_transitions = uncalibrated_transitions
                )
            transitions_by_period[period] = transitions

        # Iterate
        iterated_population = apply_transition_matrix(
            population = population.query('period == @period').copy(),
            transition_matrix = transitions,
            age_min = age_min,
            age_max_cale = age_max_cale
            )
        check_67_and_over(iterated_population, age_min = age_min + 2)
        iterated_population = add_lower_age_population(population = iterated_population, age_min = age_min, prevalence_survey = prevalence_survey)
        population = pd.concat([population, iterated_population], sort = True)

    return population, transitions_by_period


def get_initial_population(age_min = None, period = None, rescale = True, prevalence_survey = None):
    """
        Produce intiial population with disabiliyt state

        :param int age_min: minimal age to retain
        :param bool rescale: rescale using INSEE population
        :param str prevalence_survey: survey used to compute initial prevalence, should be 'care', 'hsm' or 'hsm_hsi'
    """
    assert age_min is not None
    assert prevalence_survey is not None
    if rescale:
        assert period is not None
    data_by_sex = dict()
    for sex in ['male', 'female']:
        sexe = 'homme' if sex == 'male' else 'femme'
        config = Config()
        filename = os.path.join(
            config.get('til', 'input_dir'),
            'dependance_initialisation_level_{}_{}.csv'.format(prevalence_survey, sexe)
            )
        log.info('Loading initial population dependance states from {}'.format(filename))
        df = (pd.read_csv(filename, names = ['age', 0, 1, 2, 3], skiprows = 1)
            .query('(age >= @age_min)')
            )
        df['age'] = df['age'].astype('int')

        df = (
            pd.melt(
                df,
                id_vars = ['age'],
                value_vars = [0, 1, 2, 3],
                var_name = 'initial_state',
                value_name = 'population'
                )
            .sort_values(['age', 'initial_state'])
            .set_index(['age', 'initial_state'])
            )
        assert (df.query('initial_state == -1')['population'] == 0).all()
        data_by_sex[sex] = (df
            .assign(sex = sex)
            )
    data = pd.concat(data_by_sex.values()).reset_index()

    if rescale:
        insee_population = get_insee_projected_population()
        assert period in insee_population.reset_index().year.unique()
        rescaled_data = (
            data.groupby(['sex', 'age'])['population'].sum().reset_index()
            .merge(
                (insee_population.query("year == @period")
                    .reset_index()
                    .rename(columns = {"population": "insee_population"})
                    )
                    ,
                how = 'left',
                )
            .eval("calibration = insee_population / population", inplace = False)
            )

        data = (data
            .merge(rescaled_data[['sex', 'age', 'calibration']])
            .eval("population = calibration *population", inplace = False)
            )[['sex', 'age', 'initial_state', 'population']].fillna(0).copy()

    assert data.notnull().all().all(), data.notnull().all()

    return data


def add_lower_age_population(population = None, age_min = None, prevalence_survey = None):
    """
        Ajoute dans la base des personnes aux âges les plus bas
    """
    assert age_min is not None
    assert population is not None
    assert len(population.period.unique().tolist()) == 1, 'More than one period are present: {}'.format(
        population.period.unique().tolist())
    period = population.period.unique().tolist()[0]
    lower_age_population = (get_initial_population(age_min = age_min, rescale = True, period = period, prevalence_survey = prevalence_survey)
        .query('age in [@age_min, @age_min + 1]')
        )
    lower_age_population['period'] = period
    completed_population = pd.concat(
        [lower_age_population, population],
        sort = True,
        ).sort_values(
            ['period', 'age', 'sex', 'initial_state']
            )

    assert completed_population.notnull().all().all(), completed_population.notnull().all()

    assert completed_population.notnull().all().all(), 'Missing values are present: {}'.format(
        completed_population.loc[completed_population.isnull()])

    return completed_population


def correct_transitions_for_mortality(transitions, dependance_initialisation = None, mu = None, period = None,
        survival_gain_cast = None, previous_mortality = None):
    """
        Take a transition matrix = mortality_calibrated_target and correct transitions to match period's mortality target
        according to a scenario defined by survival_gain_cast and mu
    """
    log.debug("Entering correct_transitions_for_mortality")
    death_state = 4
    assert dependance_initialisation is not None
    admissible_survival_gain_cast = ["homogeneous", "initial_vs_others", 'autonomy_vs_disability']
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
        #.rename(columns = {'mortality_insee': 'target_mortality'})
        .rename(columns = {'mortalite_2_year_insee': 'target_mortality_2_year'})
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
        .eval('correction_coefficient = target_mortality_2_year / mortality', inplace = False)
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

    return(uncalibrated_probabilities)

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
        log.debug("Using homogeneous scenario")
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
        log.debug("Using initial_vs_others scenario")
        assert mu is not None
        # Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
        other_transitions = initial_vs_others(
            period = period,
            target_mortality = mortality,
            mu = mu,
            uncalibrated_probabilities = uncalibrated_probabilities
            )
    elif survival_gain_cast == 'autonomy_vs_disability':
        log.debug("Using autonomy_vs_disability")
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


# Tools

def check_67_and_over(population, age_min):
    period = population.period.max()
    insee_population = get_insee_projected_population()
    pop_insee = insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum()
    pop_sim = population.query('(age >= @age_min) and (period == @period)')['population'].sum()
    pop_insee_62 = insee_population.query('(age == @age_min) and (year == @period)')['population'].sum()
    pop_sim_62 = population.query('(age == @age_min) and (period == @period)')['population'].sum()
    pop_insee_80 = insee_population.query('(age == @age_min + 18) and (year == @period)')['population'].sum()
    pop_sim_80 = population.query('(age == @age_min + 18) and (period == @period)')['population'].sum()
    pop_insee_100 = insee_population.query('(age == @age_min + 38) and (year == @period)')['population'].sum()
    pop_sim_100 = population.query('(age == @age_min + 38) and (period == @period)')['population'].sum()
    pop_insee_60 = insee_population.query('(age == @age_min - 2) and (year == @period)')['population'].sum()
    pop_sim_60 = population.query('(age == @age_min - 2) and (period == @period)')['population'].sum()

    log.info("period {}: insee = {} vs {} = til".format(
        period,
        insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum(),
        population.query('(age >= @age_min) and (period == @period)')['population'].sum()
        ))
    log.info("period {}: insee - til = {}".format(
        period,
        pop_insee - pop_sim
        ))
    log.info("period {}: insee_60:{} til_60:{} insee_60 - til_60 = {}".format(
       period, pop_insee_60, pop_sim_60,
       pop_insee_60 - pop_sim_60
       ))
    log.info("period {}: insee_62:{} til_62:{} insee_62 - til_62 = {}".format(
       period, pop_insee_62, pop_sim_62,
       pop_insee_62 - pop_sim_62
       ))
    log.info("period {}: insee_80:{} til_80:{} insee_80 - til_80 = {}".format(
       period, pop_insee_80, pop_sim_80,
       pop_insee_80 - pop_sim_80
       ))
    log.info("period {}: insee_100:{} til_100:{} insee_100 - til_100 = {}".format(
       period, pop_insee_100, pop_sim_100,
       pop_insee_100 - pop_sim_100
    ))


def regularize(transition_matrix_dataframe = None, by = None, probability = None, delta = None):
    assert transition_matrix_dataframe is not None
    assert by is not None
    assert probability is not None
    assert delta is not None
    final_state = 4
    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    mortality_transitions = transition_matrix_dataframe.query('final_state == @final_state').copy()

    # by_without_initial_state = [by_value for by_value in by if by_value != 'initial_state']
    problematic_indices = (mortality_transitions[probability]
        .loc[mortality_transitions[probability] >= (1 - delta)]
        .reset_index()
        .drop(['final_state', probability], axis = 1)
        )
    count = (problematic_indices
        .merge(
            transition_matrix_dataframe.reset_index()[by + ['final_state']])
        .query('final_state != @final_state')
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
        (corrected_transition_matrix.final_state == final_state),
        probability
        ] = (1 - delta)
    corrected_transition_matrix.loc[
        corrected_transition_matrix.final_state != final_state,
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
        transition_matrix_dataframe.query('final_state == @final_state')[probability] < 1
        ).all(), "There are {} 1's in transition_matrix_dataframe.{}".format(
            (transition_matrix_dataframe.query('final_state == @final_state')[probability] < 1).sum(),
            probability,
            )

    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    return transition_matrix_dataframe


def apply_transition_matrix(population = None, transition_matrix = None, age_min = None, age_max_cale = None):
    assert age_min is not None
    death_state = 4
    assert population is not None and transition_matrix is not None
    assert len(population.period.unique()) == 1
    final_population = (population
        #.loc[population['age']<age_max_cale]
        .merge(
            transition_matrix.reset_index().drop('period', axis = 1),
            on = ['age', 'sex', 'initial_state'])
        .eval('new_population = population * calibrated_probability', inplace = False)
        .drop(['initial_state', 'calibrated_probability', 'population'], axis = 1)
        .rename(columns = {'new_population': 'population'})
        .groupby(['age', 'sex', 'period', 'final_state'])['population'].sum()
        .reset_index()
        .rename(columns = {'final_state': 'initial_state'})
        )

    simulated_mortality = (final_population
        .query('(initial_state == @death_state) & (age <= 120) & (age >= @age_min)')
        .groupby(['sex', 'age'])['population']
        .sum() / final_population
        .query('(age <= 120) & (age >= @age_min)')
        .groupby(['sex', 'age'])['population']
        .sum()
        ).reset_index()

    period = population.period.unique()[0]

    mortality=get_insee_projected_mortality().query('(year == @period) and (age >= @age_min)').reset_index().eval('two_year_mortality=mortalite_2_year_insee')

    log.debug(simulated_mortality.merge(mortality).query("sex == 'male'").head(50))
    log.debug(simulated_mortality.merge(mortality).query("sex == 'female'").head(50))

    final_population = (final_population
        .eval('age = age + 2', inplace = False)
        .eval('period = period + 2', inplace = False)
        .query('(initial_state != @death_state) & (age <= 120)')
        .copy()
        )
    assert final_population.age.max() <= 120
    assert final_population.age.min() >= age_min + 2, "population minimal age is {} instead of {}".format(
        final_population.age.min(), age_min + 2)
    return final_population


def build_suffix(survival_gain_cast = None, mu = None, vagues = None, prevalence_survey = None):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}_'.format(mu)
    if vagues is not None:
        suffix += slugify.slugify(str(vagues), separator = "_")
    if prevalence_survey is not None:
        suffix += '_{prevalence_survey}_'.format(prevalence_survey = prevalence_survey)

    return suffix
