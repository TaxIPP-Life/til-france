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
logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)

# Paths

til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )

assets_path = os.path.join(
    til_france_path,
    'model',
    'options',
    'dependance_RT',
    'assets',
    )


config = Config()
data_path = os.path.join(
    config.get('raw_data', 'share'),
    #'share_data_for_microsimulation.csv', ##A modifier si on veut modifier la base transition
    'share_data_for_microsimulation_newvar.csv',
    )

assert os.path.exists(data_path)



from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    compute_prediction, 
    build_estimation_sample, 
    get_clean_share, 
    estimate_model
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

####


df = get_clean_share(extra_variables = extra_variables)
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
variables = ['age', 'final_state', 'married_','children_0','children_1','children_2','children_3plus']
extra_variables = ['age','married_','children_0','children_1','children_2','children_3plus']


sample = build_estimation_sample(initial_state, sex = sex, variables = variables, vagues = vagues)

formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'

result, formatted_params = estimate_model(initial_state, formula, sex = sex, variables = variables)

####
def get_clean_share(extra_variables = None):
    """
        Get SHARE relevant data free of missing observations
    """
    death_state = 4
    if extra_variables is None:
        extra_variables = list()
    df = pd.read_csv(data_path)

    log.debug("Share data contains the following variables: {}".format(df.columns))

    renaming = {
        'scale': 'initial_state',
        'mergeid': 'id',
        'year_int': 'year',
        'age_int': 'age',
        }
    assert set(renaming.keys()) < set(df.columns)
    df = (df
        .rename(columns = renaming)
        .copy()
        )
    df['sexe'] = df.male + 2 * (df.male == 0)
    del df['male']
    variables = ['id', 'initial_state', 'sexe', 'year', 'age', 'vague']

    if extra_variables:
        assert isinstance(extra_variables, list)
        variables = list(set(['id', 'initial_state', 'sexe', 'year', 'age', 'vague']).union(set(extra_variables)))
        
    df = df.reset_index(drop = True)
    #df = df.set_index(variables)

    df = df[variables].copy().dropna()  # TODO Remove the dropna
    assert df.notnull().all().all(), \
        "Some columns contains NaNs:\n {}".format(df.isnull().sum())

    for extra_variable in extra_variables:
        assert df[extra_variable].notnull().all(), "Variable {} contains null values. {}".format(
            extra_variable, df[extra_variable].value_counts(dropna = False))

    assert df.sexe.isin([1, 2]).all()

    filtered = df.dropna()
    log.debug("There are {} missing observation of initial_state out of {}".format(df.initial_state.isnull().sum(), len(df)))

    log.debug("There are {} valid observations out of {} for the following variables {}".format(
        len(filtered), len(df), df.columns))

    assert (filtered.isnull().sum() == 0).all()

    filtered["sexe"] = filtered["sexe"].astype('int').astype('category')
    filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')

    return filtered