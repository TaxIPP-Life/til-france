# -*- coding:utf-8 -*-


import os
import shutil

from ipp_macro_series_parser.demographie.build_parameters import run_all


def test_build_parameters():
    tmp_directory = os.path.abspath('./tmp')
    run_all(input_dir = tmp_directory, uniform_weight = 200)
    shutil.rmtree(tmp_directory)
