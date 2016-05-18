# -*- coding: utf-8 -*-

import logging
import numpy
import os
import pandas
import patsy
import pkg_resources
import statsmodels.api as sm

from til_core.config import Config

from openfisca_survey_manager.survey_collections import SurveyCollection


log = logging.getLogger(__name__)

config_files_directory = os.path.join(
    pkg_resources.get_distribution('openfisca-survey-manager').location)


def compute_nans_and_missing_values(dataframe, column):
        nans = dataframe[column].isnull().sum()
        missing = dataframe[column].max()
        import re
        if re.match(r'9+', str(missing)) is not None:
            missing_values = (dataframe[column] == missing).sum()
        else:
            missing = None
            missing_values = 0
        if nans != 0 or missing_values != 0:
            message = 'Variable {} has {} nans and {} missing values (= {})'.format(
                column, nans, missing_values, int(round(missing)) if missing is not None else None)
            print(message)
        return nans, missing_values, missing


def list_variables_with_missing_values(dataframe):
    for column in dataframe.columns:
        compute_nans_and_missing_values(dataframe, column)


def load_dataframe():
    # set seed
    numpy.random.seed(12345)

    # load data
    survey_collection = SurveyCollection.load(
        collection = 'hid', config_files_directory = config_files_directory)

    survey = survey_collection.get_survey('hid_domicile_1999')
    modb = survey.get_values(table = 'modb_c')
    ages = [column for column in modb.columns if column.endswith('a')]
    modb2 = modb[ages].copy()

    modb2.btoi1a.value_counts(dropna=False)
    (modb2.btoi1a == 999).sum()


def save(dataframe):
    config = Config()
    hsi_data_directory = config.get('raw_data', 'hsi_data_directory')
    dataframe.to_hdf(os.path.join(hsi_data_directory, "hsi_extract.h5"), 'individus_institutions')


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    initial_dataframe = load_dataframe()
