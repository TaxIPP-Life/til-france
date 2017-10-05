# -*- coding: utf-8 -*-

import glob
import numpy as np
import pandas as pd
import os
import sys

from til_core.config import Config


def grab_data(option = None, filename = None):
    simulation_output_dir = Config().get('til', 'output_dir')

    data_dir = os.path.join(
        simulation_output_dir,
        option
        )

    assert os.path.exists(data_dir)

    if filename is None:
        candidate_files = [fichier for fichier in glob.glob(data_dir + "/*.h5")]

        if len(candidate_files) != 1:
            print('Error: h5 file not unique')
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

    return(df_by_key)


def grab_series(dataframe = None, variable = None, agregation = np.mean, condition_str = None):
    """
    Returns a pd.Series containing agregation(variable) for
    a variable of the in the dataframe
    """

    if 'PERIOD' in dataframe:
        dataframe.rename(columns={'PERIOD': 'period'}, inplace=True)
    elif 'period' not in dataframe:
        print('Period column is missing')
        sys.exit(-1)

    if condition_str is not None:
        dataframe = dataframe[eval(condition_str)]

    dataframe = dataframe[['period', variable]]

    dataframe = dataframe.groupby('period').aggregate(agregation)

    return(dataframe)


def basic_tests(df = None, bounds = None, bounds_on_growth = None, cales = None):
    # rewrite without iloc
    # create 2 cases : one upper/lower bound, and one upper/lower bound for each period

    if bounds is not None:
        lower, upper = bounds
        failed_bound_check = (df.iloc[:, 0] < lower).any() or (df.iloc[:, 0] > upper).any()
        if failed_bound_check:
            print("Failed bound check")
            # add name of variable + period of fail
            return(False)

    if bounds_on_growth is not None:
        df_growth = df.copy().pct_change()
        lower, upper = bounds_on_growth
        failed_growth_bound_check = (df_growth.iloc[:, 0] < lower).any() or (df_growth.iloc[:, 0] > upper).any()
        if failed_growth_bound_check:
            print("Failed growth bound check")
            # add name of variable + period of fail
            return(False)

    if cales is not None:
        # put tolerance as an input
        tolerance = 10
        failed_cales_check = (abs(df.iloc[:, 0] - cales) > df.iloc[:, 0] * tolerance).any()
        if failed_cales_check:
            print("Failed cales check")
            # add name of variable + period of fail
            return(False)

    return(True)


def main():
    simulation_data_by_key = grab_data(option = 'dependance')
    print('data ok')
    test_salaire = grab_series(
        dataframe = simulation_data_by_key['/entities/individus'],
        variable = 'salaire_imposable',
        agregation = lambda x: np.percentile(x, 10),
        condition_str = "dataframe['salaire_imposable'] > 0"
        )

    print(test_salaire)

    assert basic_tests(
        df = test_salaire,
        bounds = (0, 20000),
        bounds_on_growth = (-0.1, 0.1),
        cales = 15000,
        )


if __name__ == '__main__':
    # sys.exit(main())
    main()
