# -*- coding:utf-8 -*-


import os
import shutil

from ipp_macro_series_parser.demographie.build_parameters import run_all


def test_build_parameters():
    tmp_directory = os.path.abspath('./tmp')
    input = os.path.abspath('../til-france//til_france/param/demo')
    run_all(pop_input_dir = input, parameters_dir = tmp_directory, uniform_weight = 200)
    shutil.rmtree(tmp_directory)
