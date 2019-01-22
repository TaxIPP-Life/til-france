# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:49:46 2019

@author: a.rain
"""
from __future__ import division


import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import slugify
import sys
import pkg_resources
import ipdb
import numpy
import statsmodels.formula.api as smf
import patsy

from til_core.config import Config

log = logging.getLogger(__name__)

logging.basicConfig(level = logging.WARNING, stream = sys.stdout)


from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    compute_prediction, 
    build_estimation_sample, 
    get_clean_share
    )

def estimate_model(initial_state, formula, sex = None, variables = ['age', 'final_state']):
    assert (sex in ['male', 'female']) or (sex is None)
    final_states = final_states_by_initial_state[initial_state]
    sample = build_estimation_sample(initial_state, sex = sex, variables = variables)
    result = smf.mnlogit(
        formula = formula,
        data = sample[variables],
        ).fit()
    log.debug(result.summary())
    formatted_params = result.params.copy()
    formatted_params.columns = sorted(set(final_states))[1:]

    def rename_index_func(index):
        index = index.lower()
        if index.startswith('i('):
            index = index[1:]
        elif index.startswith('intercept'):
            index = '(age > 0)'  # Hack to deal with https://github.com/pandas-dev/pandas/issues/16363
        return index

    formatted_params.rename(index = rename_index_func, inplace = True)
    formatted_params[sorted(set(final_states))[0]] = 0

    return result, formatted_params

##Prediction initiale
formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
vagues = [4,5,6]

uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues, estimation_survey = 'share')

##Nouvelle prediction
formula_new = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3) + children_0 + children_1 + children_2 + children_3plus + partnerinhh'
vagues = [4,5,6]

final_states_by_initial_state = {
  0: [0, 1, 4],
  1: [0, 1, 2, 4],
  2: [1, 2, 3, 4],
  3: [2, 3, 4],
  }

uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues, estimation_survey = 'share')

test = build_estimation_sample(initial_state = 0, sex = 'male', variables = ['age'], readjust = False, vagues = vagues, estimation_survey = 'share')

def compute_prediction2(initial_state = 0, formula = formula, variables = ['age'], exog = None, sex = 'male', vagues = vagues, estimation_survey = 'share'):
    """
    Compute prediction on exogneous if given or on sample
    """
    assert estimation_survey is not None
    assert initial_state is not None
    assert (sex in ['male', 'female']) or (sex is None)
    sample = build_estimation_sample(initial_state, sex = sex, variables = variables, vagues = vagues, estimation_survey = estimation_survey)
    if exog is None:
        exog = sample[variables]

    result = smf.mnlogit(
        formula = formula,
        data = sample,
        ).fit()
    #print result
    expurged_formula = formula.split('~', 1)[-1]
    #print expurged_formula
    x = patsy.dmatrix(expurged_formula, data= exog)  # exog is data for prediction
    #print x
    prediction = result.predict(x, transform=False)
    print prediction
    (abs(prediction.sum(axis = 1) - 1) < .00001).all()
    prediction = pd.DataFrame(prediction)
    final_states = final_states_by_initial_state[initial_state]
    prediction.columns = ['proba_etat_{}'.format(state) for state in sorted(final_states)]
    return prediction.reset_index(drop = True)

resultat_pred = compute_prediction2(initial_state = 0, formula = formula, variables = ['age'], exog = None, sex = 'male', vagues = vagues, estimation_survey = 'share')

def build_estimation_sample(initial_state, sex = None, variables = None, vagues = None):
    """
        Build estimation sample from share data
    """
    assert estimation_survey is not None
    final_states = final_states_by_initial_state[initial_state]
    assert (sex in ['male', 'female']) or (sex is None)
    extra_variables = None
    if variables is not None:
        extra_variables = [variable for variable in variables if variable not in ['final_state']]


    clean_share = get_clean_share(extra_variables = extra_variables)

    assert clean_share.notnull().all().all()

    assert initial_state in final_states
    no_transition = (clean_share
        .groupby('id')['initial_state']
        .count() == 1
        )
    no_transition_with_specific_initial_state = clean_share.loc[
        clean_share.id.isin(no_transition.index[no_transition].tolist())
        ].query('initial_state == {}'.format(initial_state))
    log.info(
        "There are {} individuals out of {} with only one observation (no transition) with intiial state = {}".format(
            len(no_transition_with_specific_initial_state), no_transition.sum(), initial_state))

    clean_share['final_state'] = clean_share.groupby('id')['initial_state'].shift(-1).copy()
    log.info("There are {} individuals with intiial state = {} with no subsequent transition".format(
        len(
            clean_share.loc[clean_share.final_state.isnull()]
            .query('(initial_state == {})'.format(initial_state))
            ),
        initial_state,
        ))
    log.debug("Transifions from initial_state = {}:\n {}".format(
        initial_state,
        (
            clean_share
            .query('(initial_state == {})'.format(initial_state))['final_state']
            .value_counts(dropna = False)
            .sort_index()
            )
        ))
    wrong_transition = (clean_share.loc[clean_share.final_state.notnull()]
        .query('(initial_state == {}) & (final_state not in {})'.format(
            initial_state, final_states)
            )
        .groupby('final_state')['initial_state']
        .count()
        )
    log.info("There are {} individuals with intiial state = {} transiting to forbidden states: \n {}".format(
        wrong_transition.sum(), initial_state, wrong_transition[wrong_transition > 0]))

    log.info("Final states:\n {}".format(clean_share.final_state.value_counts(dropna = False)))

    if wrong_transition.sum() > 0:
        if initial_state in replace_by_initial_state.keys():
            log.info("Using the following replacement rule: {}".format(
                replace_by_initial_state[initial_state]
                ))
            log.debug("Sample size before cleaning bad final_states:\n {}".format(
                len(clean_share
                    .query('(initial_state == {})'.format(
                        initial_state,
                        ))
                    .dropna()
                    )
                ))
            log.info(clean_share.query('(initial_state == {})'.format(
                initial_state,
                )).final_state.value_counts(dropna = False))
            sample = (clean_share
                .query('initial_state == {}'.format(initial_state))
                .dropna()
                .replace({
                    'final_state': replace_by_initial_state[initial_state]
                    })
                .copy()
                )
            log.debug("Sample size after cleaning bad final_states:\n {}".format(len(sample)))
            log.info(sample.final_state.value_counts())
    else:
        sample = (clean_share
            .query('(initial_state == {})'.format(
                initial_state,
                ))
            .dropna()
            .copy()
            )

    log.debug("Sample size after eventually cleaning bad final_states {}".format(len(sample)))

    if sex:
        if sex == 'male':
            sample = sample.query('sexe == 1').copy()
        elif sex == 'female':
            sample = sample.query('sexe == 2').copy()
    sample["final_state"] = sample["final_state"].astype('int').astype('category')

    if sex:
        log.info("Keeping sample of size {} for sex = {}".format(len(sample), sex))
    else:
        log.info("Keeping sample of size {}".format(len(sample)))
        del sample['sexe']

    if vagues:
        sample = sample.query('vague in @vagues').copy()

    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states), '{} differs from {}'.format(
        set(sample.final_state.value_counts().index.tolist()), set(final_states)
        )
    return sample.reset_index(drop = True).copy()


df = get_clean_share()
    sex = 'male'
    initial_state = 1
    estimation_survey = 'share'
    vagues = [4,5,6]
final_states_by_initial_state = {
  0: [0, 1, 4],
  1: [0, 1, 2, 4],
  2: [1, 2, 3, 4],
  3: [2, 3, 4],
  }


    sample = build_estimation_sample(initial_state, sex = sex, vagues = vagues)

    sex = None
    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'
    variables = ['age', 'final_state']

    initial_state = 0
    result, formatted_params = estimate_model(initial_state, formula, sex = sex, variables = variables)

def build_estimation_sample(initial_state, sex = None, variables = None, vagues = None):
    """
        Build estimation sample from share data
    """
    #assert estimation_survey is not None
    final_states = final_states_by_initial_state[initial_state]
    assert (sex in ['male', 'female']) or (sex is None)
    extra_variables = None
    if variables is not None:
        extra_variables = [variable for variable in variables if variable not in ['final_state']]


    clean_share = get_clean_share(extra_variables = extra_variables)

    assert clean_share.notnull().all().all()

    assert initial_state in final_states
    no_transition = (clean_share
        .groupby('id')['initial_state']
        .count() == 1
        )
    no_transition_with_specific_initial_state = clean_share.loc[
        clean_share.id.isin(no_transition.index[no_transition].tolist())
        ].query('initial_state == {}'.format(initial_state))
    log.info(
        "There are {} individuals out of {} with only one observation (no transition) with intiial state = {}".format(
            len(no_transition_with_specific_initial_state), no_transition.sum(), initial_state))

    clean_share['final_state'] = clean_share.groupby('id')['initial_state'].shift(-1).copy()
    log.info("There are {} individuals with intiial state = {} with no subsequent transition".format(
        len(
            clean_share.loc[clean_share.final_state.isnull()]
            .query('(initial_state == {})'.format(initial_state))
            ),
        initial_state,
        ))
    log.debug("Transifions from initial_state = {}:\n {}".format(
        initial_state,
        (
            clean_share
            .query('(initial_state == {})'.format(initial_state))['final_state']
            .value_counts(dropna = False)
            .sort_index()
            )
        ))
    wrong_transition = (clean_share.loc[clean_share.final_state.notnull()]
        .query('(initial_state == {}) & (final_state not in {})'.format(
            initial_state, final_states)
            )
        .groupby('final_state')['initial_state']
        .count()
        )
    log.info("There are {} individuals with intiial state = {} transiting to forbidden states: \n {}".format(
        wrong_transition.sum(), initial_state, wrong_transition[wrong_transition > 0]))

    log.info("Final states:\n {}".format(clean_share.final_state.value_counts(dropna = False)))

    if wrong_transition.sum() > 0:
        if initial_state in replace_by_initial_state.keys():
            log.info("Using the following replacement rule: {}".format(
                replace_by_initial_state[initial_state]
                ))
            log.debug("Sample size before cleaning bad final_states:\n {}".format(
                len(clean_share
                    .query('(initial_state == {})'.format(
                        initial_state,
                        ))
                    .dropna()
                    )
                ))
            log.info(clean_share.query('(initial_state == {})'.format(
                initial_state,
                )).final_state.value_counts(dropna = False))
            sample = (clean_share
                .query('initial_state == {}'.format(initial_state))
                .dropna()
                .replace({
                    'final_state': replace_by_initial_state[initial_state]
                    })
                .copy()
                )
            log.debug("Sample size after cleaning bad final_states:\n {}".format(len(sample)))
            log.info(sample.final_state.value_counts())
    else:
        sample = (clean_share
            .query('(initial_state == {})'.format(
                initial_state,
                ))
            .dropna()
            .copy()
            )

    log.debug("Sample size after eventually cleaning bad final_states {}".format(len(sample)))

    if sex:
        if sex == 'male':
            sample = sample.query('sexe == 1').copy()
        elif sex == 'female':
            sample = sample.query('sexe == 2').copy()
    sample["final_state"] = sample["final_state"].astype('int').astype('category')

    if sex:
        log.info("Keeping sample of size {} for sex = {}".format(len(sample), sex))
    else:
        log.info("Keeping sample of size {}".format(len(sample)))
        del sample['sexe']

    if vagues:
        sample = sample.query('vague in @vagues').copy()

    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states), '{} differs from {}'.format(
        set(sample.final_state.value_counts().index.tolist()), set(final_states)
        )
    return sample.reset_index(drop = True).copy()

