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