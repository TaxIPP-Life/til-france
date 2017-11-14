#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division


import argparse
import logging
import numpy
import os
import pandas
import pkg_resources
import sys


from til_france.targets import population

app_name = os.path.splitext(os.path.basename(__file__))[0]
log = logging.getLogger(app_name)


def build_dataframe(year = 2010, weight_threshold = 200):
    sexe_value_by_gender = dict(male = 0, female = 1)
    dataframe_by_gender = dict()
    for gender, sexe_value in sexe_value_by_gender.iteritems():
        dataframe = population.get_data_frame_insee(gender = gender, by = 'age')
        dataframe.reset_index(inplace = True)
        dataframe = dataframe[['age', year]]
        dataframe['period'] = year
        dataframe.rename(columns = {year: 'weight'}, inplace = True)

        low_weight_dataframe = dataframe.query('weight < @weight_threshold')
        weights_of_random_picks = low_weight_dataframe.weight.sum()
        number_of_picks = int(weights_of_random_picks / weight_threshold)
        log.info('Extracting {} from {} observations with weights lower than {} representing {} individuals'.format(
            number_of_picks,
            low_weight_dataframe.weight.count(),
            weight_threshold,
            weights_of_random_picks
            ))
        sample = low_weight_dataframe.sample(n = number_of_picks, weights = 'weight', random_state = 12345)
        sample.weight = weight_threshold
        dataframe = dataframe.query('weight >= @weight_threshold').append(sample)
        dataframe['n'] = (dataframe.weight / weight_threshold).round().astype(int)
        dataframe = dataframe.loc[numpy.repeat(dataframe.index.values, dataframe.n)]
        dataframe = dataframe[['age', 'period']].copy()
        dataframe['sexe'] = sexe_value
        dataframe_by_gender[gender] = dataframe

    result = pandas.concat(dataframe_by_gender.values(), ignore_index = True)
    result.index.name = 'id'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action = 'store_true', default = False, help = "force overwrite")
    parser.add_argument('-o', '--output', default = None, help = "path to output csv file")
    parser.add_argument('-v', '--verbose', action = 'store_true', default = False, help = "increase output verbosity")
    parser.add_argument('-w', '--weight', default = 200, help = "uniform weight of the resulting population")
    parser.add_argument('-y', '--year', default = 2010, help = "year of the population distribution")
    args = parser.parse_args()
    logging.basicConfig(level = logging.DEBUG if args.verbose else logging.WARNING, stream = sys.stdout)
    file_path = args.output
    dataframe = build_dataframe(year = args.year, weight_threshold = args.weight)
    if file_path is None:
        file_path = os.path.join(
            pkg_resources.get_distribution('til-france').location,
            'til_france',
            'tests',
            'demographic_model',
            'input',
            'individus.csv'
            )
    file_path = os.path.abspath(file_path)
    if os.path.exists(file_path):
        if not args.force:
            log.info('File {} already exists. Use option -f to overwrite.'.format(file_path))
        else:
            log.info('Writing data in {}.'.format(file_path))
            dataframe.to_csv(file_path)
    else:
        log.info('Path {} does not exists, creating it'.format(file_path))
        dirs_to_be_created = [os.path.dirname(file_path)]
        while not os.path.exists(dirs_to_be_created[-1]):
            dirs_to_be_created.append(os.path.dirname(dirs_to_be_created[-1]))
        for dir in dirs_to_be_created[1::-1]:
            print(dir)
            os.mkdir(dir)
        dataframe.to_csv(file_path)

if __name__ == "__main__":
    sys.exit(main())
