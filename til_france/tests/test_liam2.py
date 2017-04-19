# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pkg_resources

from liam2.simulation import Simulation
from liam2.importer import csv2h5


# TODO mutualize with liam2
use_travis = os.environ.get('USE_TRAVIS', None) == 'true'

test_root = os.path.join(
    pkg_resources.get_distribution('til_france').location,
    'til_france',
    'tests',
    'models',
    )


def run_file(test_file):
    print(test_file)
    if 'import' in test_file:
        print("Importing", test_file)
        csv2h5(test_file)
    else:
        output_dir = os.path.join(os.path.dirname(test_file), 'output')
        print('Running {} using {} as output dir'.format(test_file, output_dir))
        simulation = Simulation.from_yaml(test_file, output_dir=output_dir)
        simulation.run()


def iterate_directory(directory, dataset_creator, excluded_files):
    directory_path = os.path.join(test_root, directory)
    if dataset_creator:
        excluded_files = excluded_files + (dataset_creator,)
        yield os.path.join(directory_path, dataset_creator)
    for test_file in os.listdir(directory_path):
        if test_file.endswith('.yml') and test_file not in excluded_files:
            yield os.path.join(directory_path, test_file)


def test_demographic_model():
    excluded = ()
    # if use_travis:
    #     excluded += None
    for test_file in iterate_directory('demographic_model', 'import.yml', excluded):
        yield run_file, test_file


def test_maximize_utility_model():
    excluded = ()
    for test_file in iterate_directory('maximize_utility', None, excluded):
        yield run_file, test_file


if __name__ == '__main__':
    for test in test_maximize_utility_model():
        func, args = test[0], test[1:]
        func(*args)
