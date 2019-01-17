# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:51:55 2018

@author: a.rain
"""

import logging
import numpy as np
import os
import pandas as pd
#import pkg_resources
import seaborn as sns
import sys

from til_france.model.options.dependance_RT.life_expectancy.share.simulation_all import (
    get_insee_projected_mortality,
    )



from til_core.config import Config
from til_france.tests.base import ipp_colors
from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import get_clean_paquid


colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


#figures_directory = os.path.join(
#    pkg_resources.get_distribution('til-france').location,
#    'til_france',
#    'figures'
#    )


def smooth_pivot_table(pivot_table, window = 7, std = 2):
    smoothed_pivot_table = pivot_table.copy()
    for dependance_niveau in smoothed_pivot_table.columns:
        smoothed_pivot_table[dependance_niveau] = (pivot_table[dependance_niveau]
            .rolling(win_type = 'gaussian', center = True, window = window, axis = 0)
            .mean(std = std)
            )

    return smoothed_pivot_table

def check_67_and_over(population, age_min):
    period = population.period.max()
    insee_population = get_insee_projected_population()
    pop_insee = insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum()
    pop_sim = population.query('(age >= @age_min) and (period == @period)')['population'].sum()
    log.info("period {}: insee = {} vs {} = til".format(
        period,
        insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum(),
        population.query('(age >= @age_min) and (period == @period)')['population'].sum()
        ))
    log.info("period {}: insee - til = {}".format(
        period,
        pop_insee - pop_sim   
))

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


def apply_transition_matrix(population = None, transition_matrix = None, age_min = None):
    assert age_min is not None
    death_state = 4
    assert population is not None and transition_matrix is not None
    assert len(population.period.unique()) == 1
    final_population = (population
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
    mortality = get_insee_projected_mortality().query('(year == @period) and (age >= @age_min)').reset_index().eval(
        'two_year_mortality = 1 - (1 - mortality) ** 2', inplace = False)

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


def build_suffix(survival_gain_cast = None, mu = None, vagues = None, survey = None):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)
    if vagues is not None:
        suffix += slugify.slugify(str(vagues), separator = "_")
    if survey is not None:
        suffix += '_{survey}_'.format(survey = survey)

    return suffix


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