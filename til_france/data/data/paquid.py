# -*- coding: utf-8 -*-

import logging
import os
import pandas


from openfisca_survey_manager.survey_collections import SurveyCollection
from til_core.config import Config


def load_dataframe():
    # Load data
    survey_collection = SurveyCollection.load(collection = 'paquid')
    survey = survey_collection.get_survey('hsi_hsi_2009')


def save(dataframe):
    config = Config()
    hsi_data_directory = config.get('raw_data', 'paquid')
    dataframe.to_hdf(os.path.join(hsi_data_directory, "paquid_extract.h5"), 'paquid')
