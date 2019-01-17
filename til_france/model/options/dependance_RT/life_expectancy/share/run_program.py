# -*- coding: utf-8 -*-


from __future__ import division


import logging
import os
import sys


from til_core.config import Config


from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    )

from til_france.model.options.dependance_RT.life_expectancy.share.simulation_all import run


from til_france.model.options.dependance_RT.life_expectancy.share.paths_prog import (
    til_france_path,
    assets_path
    )


from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in [
    'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)

logging.basicConfig(level = logging.WARNING, stream = sys.stdout)

life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )
config = Config()
figures_directory = config.get('dependance', 'figures_directory')


### TEST 1 données CARE en prevalence

vagues = [1,2]
vagues = [4,5,6]


formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
uncalibrated_transitions = get_transitions_from_formula(
    formula = formula,
    vagues = vagues,
    estimation_survey = 'share'
    )

mu = 1

survival_gain_casts = [
        #'homogeneous',
        #'initial_vs_others',
        'autonomy_vs_disability'
        ]

run(
    survival_gain_casts,
    mu = mu,
    uncalibrated_transitions = uncalibrated_transitions,
    vagues = vagues,
    age_min = 60,
    prevalence_survey = 'care',
    one_year_approximation = False,
    age_max_cale = 99,
    )
