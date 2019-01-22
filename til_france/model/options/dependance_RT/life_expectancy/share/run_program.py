# -*- coding: utf-8 -*-


from __future__ import division


import logging
import sys


from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    get_transitions_from_formula,
    )

from til_france.model.options.dependance_RT.life_expectancy.share.simulation_all import run


log = logging.getLogger(__name__)
logging.basicConfig(level = logging.WARNING, stream = sys.stdout)


vagues = [1,2]
vagues = [4,5,6]


formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
uncalibrated_transitions = get_transitions_from_formula(
    formula = formula,
    vagues = vagues,
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
