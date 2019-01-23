# -*- coding: utf-8 -*-


from __future__ import division


import logging
import os
import pandas as pd
import sys
import pkg_resources

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


from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    compute_prediction,
    build_estimation_sample,
    get_clean_share,
    estimate_model
    )


####

sex = 'male'
initial_state = 0
estimation_survey = 'share'
vagues = [4,5,6]
final_states_by_initial_state = {
  0: [0, 1, 4],
  1: [0, 1, 2, 4],
  2: [1, 2, 3, 4],
  3: [2, 3, 4],
  }

variables = ['age', 'final_state', 'married_','children_0','children_1','children_2','children_3plus']

extra_variables = ['age', 'married_', 'children_0','children_1', 'children_2', 'children_3plus']


df = get_clean_share(extra_variables = extra_variables)

#sample = build_estimation_sample(initial_state, sex = sex, variables = variables, vagues = vagues)

formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'
formula2 = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3) + married_ + children_0 + children_1 + children_2 + children_3plus'

#Get estimation coefficients
result, formatted_params = estimate_model(initial_state, formula2, sex = sex, variables = variables)

print(result.summary())
print(formatted_params) 
# Get predictions
age_min = 50
age_max = 120
exog = pd.DataFrame(dict(age = range(age_min, age_max + 1)))

prediction = compute_prediction(initial_state = initial_state, formula = formula, variables = variables, exog = None, sex = sex, vagues = vagues)

print(prediction)