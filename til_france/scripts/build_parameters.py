#! /usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import logging
import os
import sys


from til_france.targets import dependance, population

app_name = os.path.splitext(os.path.basename(__file__))[0]
log = logging.getLogger(app_name)


def run_all(input_dir = None, uniform_weight = None):
    assert uniform_weight is not None
    dependance.build_prevalence_2010(input_dir = input_dir, uniform_weight = uniform_weight)
    dependance.build_prevalence_all_years(input_dir = input_dir, to_csv = True)
    population.rescale_migration(input_dir = input_dir)
    population.build_mortality_rates(input_dir = input_dir, to_csv = True)
    # Missing fecondite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--destination', default = None, help = "destination directory")
    parser.add_argument('-w', '--weight', default = 200, help = "weight used")  # TODO remove weight from here
    parser.add_argument('-v', '--verbose', action = 'store_true', default = False, help = "increase output verbosity")
    args = parser.parse_args()
    logging.basicConfig(level = logging.DEBUG if args.verbose else logging.WARNING, stream = sys.stdout)
    input_dir = os.path.abspath(args.destination)
    if not os.path.exists(input_dir):
        log.info('Creating directory {}'.format(input_dir))
        os.makedirs(input_dir)
    parameters_dir = os.path.join(input_dir, 'parameters')
    if not os.path.exists(parameters_dir):
        log.info('Creating directory {}'.format(parameters_dir))
        os.makedirs(parameters_dir)
    run_all(input_dir = input_dir, uniform_weight = args.weight)


if __name__ == "__main__":
    sys.exit(main())
