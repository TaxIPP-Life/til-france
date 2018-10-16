# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:01:48 2018

@author: a.rain
"""
import logging
import numpy as np
import os
import pandas as pd
import pkg_resources
import seaborn as sns
import sys


## Paths

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