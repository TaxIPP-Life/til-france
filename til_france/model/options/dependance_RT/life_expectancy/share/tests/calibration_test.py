# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:45:40 2018

@author: a.rain
"""


##TEST CALIBRATION

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import slugify
import sys
import patsy

from til_core.config import Config

from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    )

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    get_insee_projected_mortality,
    get_insee_projected_population,
    plot_projected_target,
    regularize
    )

from til_france.model.options.dependance_RT.life_expectancy.share.calibration import (
    build_mortality_calibrated_target_from_formula,
    build_mortality_calibrated_target_from_transitions,
    correct_transitions_for_mortality
    )

from til_france.data.data.hsm_dependance_niveau import (
    create_dependance_initialisation_share,
    get_hsi_hsm_dependance_gir_mapping,
    )


from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in [
    'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)

life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )

config = Config()

figures_directory = config.get('dependance', 'figures_directory')

#########


def correct_transitions(transitions, probability_name = 'calibrated_probability'):
    assert probability_name in transitions.columns, "Column {} not found in transitions columns {}".format(
        probability_name, transitions.columns)
    correction = False
    if correction:
        central_age = 93
        width = 2

        transitions = transitions.copy().rename(columns = {probability_name: 'calibrated_probability'})

        corrections = transitions.query('(initial_state == 3) & (final_state == 4)').copy()
        corrections.eval('factor = (1 + tanh( (age- @central_age) / @width)) / 2', inplace = True)
        corrections.eval('calibrated_probability = factor * calibrated_probability + 0 * (1 - factor)', inplace = True)

        transitions.update(corrections)

        corrections_3_3 = (1 - (
            transitions
                .query('(initial_state == 3) and (final_state != 3)')
                .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
                )).reset_index()

        corrections_3_3['final_state'] = 3
        corrections_3_3 = corrections_3_3.set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        transitions.update(corrections_3_3)
        #    transitions.query(
        #        "(period == 2012) and (sex == 'male') and (age <= 100) and (initial_state == 3) and (final_state == 3)"
        #        ).plot(y = ['calibrated_probability'])
        return transitions.rename(columns = {'calibrated_probability': probability_name})

    else:
        return transitions
 
def get_transitions_from_formula(formula = None, age_min = 50, age_max = 120, vagues = None):
    transitions = None
    for sex in ['male', 'female']:
        assert formula is not None
        proba_by_initial_state = dict()
        exog = pd.DataFrame(dict(age = range(age_min, age_max + 1)))
        for initial_state, final_states in final_states_by_initial_state.iteritems():
            proba_by_initial_state[initial_state] = pd.concat(
                [
                    exog,
                    compute_prediction(initial_state, formula, sex = sex, exog = exog, vagues = vagues)
                    ],
                axis = 1,
                )
        transitions = pd.concat([
            transitions,
            build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = sex)
            ])

    transitions.reset_index().set_index(['sex', 'age', 'initial_state', 'final_state'])
    return transitions

 ##### TEST

#Faire tourner les fonctions de simulation.py avant
vagues = [1, 2]
formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues)



   
correct_transitions(
        uncalibrated_transitions,
        probability_name = 'probability'
        )

 expurged_formula = formula.split('~', 1)[-1]
 
sample = build_estimation_sample(initial_state, sex = sex, variables = variables, vagues = vagues)

 variables = ['age']
 exog = sample[variables]
 
 x = patsy.dmatrix(expurged_formula, data= exog)  # exog is data for prediction


#####
 
 def compute_prediction(initial_state = None, formula = None, variables = ['age'], exog = None, sex = None, vagues = None):
    """
    Compute prediction on exogneous if given or on sample
    """
    assert initial_state is not None
    assert (sex in ['male', 'female']) or (sex is None)
    sample = build_estimation_sample(initial_state, sex = sex, variables = variables, vagues = vagues)
    if exog is None:
        exog = sample[variables]

    result = smf.mnlogit(
        formula = formula,
        data = sample,
        ).fit()
    expurged_formula = formula.split('~', 1)[-1]

    x = patsy.dmatrix(expurged_formula, data = exog)  # exog is data for prediction
    prediction = result.predict(x, transform=False)
    (abs(prediction.sum(axis = 1) - 1) < .00001).all()
    prediction = pd.DataFrame(prediction)
    final_states = final_states_by_initial_state[initial_state]
    prediction.columns = ['proba_etat_{}'.format(state) for state in sorted(final_states)]
    return prediction.reset_index(drop = True)

#########
    
import numpy as np
from patsy import dmatrices, dmatrix, demo_data
data = demo_data("a", "b", "x1", "x2", "y", "z column")

dmatrices("y ~ x1 + x2", data)

###
final_states_by_initial_state = {
  0: [0, 1, 4],
  1: [0, 1, 2, 4],
  2: [1, 2, 3, 4],
  3: [2, 3, 4],
  }

initial_state = 0
final_states = final_states_by_initial_state[initial_state]

get_mortality_after_imputation(mortality_table = None, dependance_initialisation = None, age_min = 50):

build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = sex)

####
vagues = [4,5,6]
formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues, estimation_survey = 'share')
