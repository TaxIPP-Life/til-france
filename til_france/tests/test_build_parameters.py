# -*- coding:utf-8 -*-


import os
import shutil

from til_france.scripts.build_parameters import run_all


def test_build_parameters():
    tmp_directory = os.path.abspath('./tmp')
    run_all(input_dir = tmp_directory, uniform_weight = None)
    shutil.rmtree(tmp_directory)
