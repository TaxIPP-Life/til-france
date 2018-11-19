# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:42:56 2018

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


from til_core.config import Config


from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    compute_prediction, 
    build_estimation_sample, 
    get_clean_share
    )

from til_france.model.options.dependance_RT.life_expectancy.share.simulation_all import (
    run,
    save_data_and_graph,
    run_scenario,
    run_scenario2,
    create_initial_prevalence,
    get_care_prevalence_pivot_table,
    get_initial_population,
    add_lower_age_population,
    build_mortality_calibrated_target_from_transitions,
    build_mortality_calibrated_target,
    _get_calibrated_transitions,
    _compute_calibration_coefficient,
    correct_transitions_for_mortality,
    get_mortality_after_imputation,
    get_predicted_mortality_table,
    get_insee_projected_mortality,
    get_insee_projected_mortality_interm,
    #get_insee_projected_mortality_next_period,
    get_insee_projected_population,
    smooth_pivot_table,
    check_67_and_over,
    regularize,
    regularize2,
    apply_transition_matrix,
    build_suffix,
    assert_probabilities,    
    impute_high_ages,
    )


#from til_france.model.options.dependance_RT.life_expectancy.share.tool_prog import (
#    smooth_pivot_table,
#    regularize,
#    check_67_and_over,
#    apply_transition_matrix,
#    build_suffix,
#    assert_probabilities
#    )

from til_france.model.options.dependance_RT.life_expectancy.share.paths_prog import (
    til_france_path,
    assets_path
    )


from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in [
    'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)

logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)

life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )

config = Config()

figures_directory = config.get('dependance', 'figures_directory')

## Paths

from til_france.model.options.dependance_RT.life_expectancy.share.paths_prog import (
    til_france_path,
    assets_path
    )

#######

### TEST 1 données CARE en prevalence

vagues = [1,2]
vagues = [4,5,6]


formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues, estimation_survey = 'share')

survival_gain_casts = [
        #'homogeneous',
        #'initial_vs_others',
        'autonomy_vs_disability'
        ]


mu = 1

run(survival_gain_casts,mu,uncalibrated_transitions = uncalibrated_transitions, vagues = vagues, age_min = 60, prevalence_survey = 'care', transformation_1an = False, age_max_cale=99)


run(survival_gain_casts,uncalibrated_transitions = uncalibrated_transitions, vagues = vagues, age_min = 60, prevalence_survey = 'care', transformation_1an = False, age_max_cale=101)

