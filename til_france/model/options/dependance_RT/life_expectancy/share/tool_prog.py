# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:51:55 2018

@author: a.rain
"""

import logging
import numpy as np
import os
import pandas as pd
import pkg_resources
import seaborn as sns
import sys


from til_core.config import Config
from til_france.tests.base import ipp_colors
from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import get_clean_paquid


colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


figures_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'figures'
    )


def smooth_pivot_table(pivot_table, window = 7, std = 2):
    smoothed_pivot_table = pivot_table.copy()
    for dependance_niveau in smoothed_pivot_table.columns:
        smoothed_pivot_table[dependance_niveau] = (pivot_table[dependance_niveau]
            .rolling(win_type = 'gaussian', center = True, window = window, axis = 0)
            .mean(std = std)
            )

    return smoothed_pivot_table