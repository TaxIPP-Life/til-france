# -*- coding: utf-8 -*-

from __future__ import division

import logging
import glob
import numpy as np
import pandas as pd
import os
import sys
import time

from til_core.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grab_data(option = None, filename = None):
    logger.info('Loading data from h5 file...')
    start = time.time()

    simulation_output_dir = Config().get('til', 'output_dir')

    data_dir = os.path.join(
        simulation_output_dir,
        option
        )

    assert os.path.exists(data_dir)

    if filename is None:
        candidate_files = [fichier for fichier in glob.glob(data_dir + "/*.h5")]

        if len(candidate_files) != 1:
            logger.warning('Error: h5 file not unique and no file name given')
            sys.exit(-1)
        else:
            data_source = candidate_files[0]
    else:
        data_source = os.path.join(
            data_dir,
            filename
            )

    data = pd.HDFStore(data_source)
    keys = data.keys()

    df_by_key = {key: data[key] for key in keys}
    data.close()
    stop = time.time()
    logger.info('... done in {}s'.format(int(stop - start)))

    return(df_by_key)


class test_on_a_series:
    def __init__(self, name = None, dataframe = None, variable = None, agregation = None,
    condition_string = None, periods = None, checks = None):
        self.name = name
        self.dataframe = dataframe
        self.variable = variable
        self.agregation = agregation
        self.condition_string = condition_string
        self.periods = periods
        self.checks = checks

    def load_data(self):
        self.dataframe = self.dataframe.copy()

        if 'PERIOD' in self.dataframe:
            self.dataframe.rename(columns={'PERIOD': 'period'}, inplace=True)
        elif 'period' not in self.dataframe:
            logger.warning('Period column is missing')
            return(None)

        if self.condition_string is not None:
            self.dataframe = self.dataframe[eval(self.condition_string)]

        self.dataframe = self.dataframe[['period', self.variable]]

        self.dataframe = self.dataframe.groupby('period') \
                                       .aggregate(self.agregation)

        if self.periods is not None:
            self.dataframe = self.dataframe.loc[list(self.periods), :]

    def perform_checks(self):
        logger.info("Starting " + self.name)

        if isinstance(self.checks, tuple):
            self.checks = [self.checks]

        passed = True

        for check in self.checks:
            check = [field for field in check]
            test_type, rest = check[0], check[1:]

            # only two types type implemented so far: bounds and sample_stat
            # bounds look if (a transformation of) the series is always within bounds eg growth rate of gdp within [-5%,5%]
            # ie check at each period a condition
            # sample_stat: some function of (a transformation of) the entire time series fits a condition eg variance of mean wage not too high
            # ie check once a condition that is a function of all periods values
            
            if test_type == 'bounds':
                transformations, minima, maxima = rest  # function or list of function, numeric or list of numeric, numeric or list of numeric
                self.temp_data = self.dataframe.copy()
                if callable(transformations):
                    # then transformation is actually just one function, not an array of functions
                    self.temp_data = transformations(self.temp_data.copy())
                elif len(transformations) == len(self.temp_data) and all([callable(k) for k in transformations]):
                    transformed_data = self.temp_data[self.variable].apply(lambda row: transformations[row.name](row))
                    self.temp_data[self.variable] = transformed_data
                else:
                    logger.warning('Length of list of transformations neither 1 nor number of rows in the dataframe')
                    sys.exit(-1)

                passed_check = (self.temp_data.iloc[:, 0] >= minima).all() and (self.temp_data.iloc[:, 0] <= maxima).all()

            elif test_type == 'sample_stat':
                formula, condition = rest
                self.temp_data = self.dataframe.copy()
                sample_stat = formula(self.temp_data.values)
                passed_check = condition(sample_stat)

            else:
                logger.warning("Wrong test type: " + test_type)
                sys.exit(-1)

            if not passed_check:
                logger.warning(self.name + " failed " + test_type + " check")
                passed = False

        return(passed)

    def run(self):
        self.load_data()
        self.all_passed = self.perform_checks()

        if self.all_passed:
            logger.info(self.name + " passed all checks")
        else:
            logger.warning(self.name + " failed at least one check of the test")


def main():
    simulation_data_by_key = grab_data(option = 'dependance')

    periods = range(2010, 2041)

    # test: are wages not too weird? eg how do quantiles of the wage distribution behave?
    def checks_on_income_percentile(k):
        checks = [
            ('bounds', lambda x: x, 15000, [100 * t + k / 100.0 * (80000 - 15000) for t in range(len(periods))]),  # test that k-th percentile of income is between 15000 (at all period) and some period-dependent value (100*t + k/100.0 * (40000 - 15000)) at all periods
            ('bounds', lambda x: x.pct_change(), -5, 5),  # test that year on year percent change of k-th percentile does not fall outside of some plausible window
            ('sample_stat', np.var, lambda x: (x > 0.1 and x < 10**5))  # there's some variation over years
            ]

        return(checks)

    tests_on_taxable_income_by_percentile = list(
        test_on_a_series(
            name = "test_salaire_centile_" + str(k*5),
            dataframe = simulation_data_by_key['/entities/individus'],
            variable = 'salaire_imposable',
            agregation = lambda x: np.percentile(x, k*5),
            condition_string = "(self.dataframe['salaire_imposable'] > 0)",
            periods = periods,
            checks = checks_on_income_percentile(k*5),
            ) for k in range(1, 20)
        )

    for percentile_test in tests_on_taxable_income_by_percentile:
        percentile_test.run()

    simple_test_on_dependance = test_on_a_series(
        name = "test_no_dependance_below_60",
        dataframe = simulation_data_by_key['/entities/individus'],
        variable = 'dependance_niveau',
        agregation = np.max,
        condition_string = "(self.dataframe['age'] < 60)",
        periods = periods,
        checks = ('bounds', lambda x: x, 0, 0),
        )

    simple_test_on_dependance.run()


if __name__ == '__main__':
    # sys.exit(main())
    main()
