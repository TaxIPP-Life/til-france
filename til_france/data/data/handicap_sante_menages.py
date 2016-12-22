# -*- coding: utf-8 -*-

import feather
import logging
import numpy
import os
import pandas as pd
import pkg_resources
import tables

from til_core.config import Config

from openfisca_survey_manager.survey_collections import SurveyCollection


log = logging.getLogger(__name__)

config_files_directory = os.path.join(
    pkg_resources.get_distributionion('openfisca-survey-manager').location)



def load_dataframe():
    # set seed
    numpy.random.seed(12345)

    # load data
    survey_collection = SurveyCollection.load(
        collection = 'hsm', config_files_directory = config_files_directory)

    survey = survey_collection.get_survey('hsm_hsm_2008')
    hsm_individus = survey.get_values(table = 'individus')

    config = Config()
    path_liam_input_data = config.get('til', 'input_dir')
    name = "patrimoine_200.h5"
    path = os.path.join(path_liam_input_data, name)
    patrimoine_individus = pd.read_hdf(path, '/entities/individus')

    path_liam_temp_data = config.get('til', 'temp_dir')
    feather.write_dataframe(hsm_individus, os.path.join(path_liam_temp_data, 'hsm.feather'))
    feather.write_dataframe(patrimoine_individus, os.path.join(path_liam_temp_data, 'patrimoine.feather'))

    return hsm_individus


def save(dataframe):
    config = Config()
    hsi_data_directory = config.get('raw_data', 'hsi_data_directory')
    dataframe.to_hdf(os.path.join(hsi_data_directory, "hsi_extract.h5"), 'individus_institutions')


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    df = load_dataframe()