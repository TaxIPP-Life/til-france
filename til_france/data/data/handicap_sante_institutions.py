# -*- coding: utf-8 -*-

import logging
import numpy
import os
import pandas
import patsy
import pkg_resources
import statsmodels.api as sm

from til_core.config import Config
from xdg import BaseDirectory

from openfisca_survey_manager.survey_collections import SurveyCollection


log = logging.getLogger(__name__)

config_files_directory = BaseDirectory.save_config_path('openfisca-survey-manager')


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
    # Set seed
    numpy.random.seed(12345)

    # Load data
    survey_collection = SurveyCollection.load(
        collection = 'hsi', config_files_directory = config_files_directory)

    survey = survey_collection.get_survey('hsi_hsi_2009')
    # CDATCO; Ancienneté de la vie en couple
    # CDATDC: Année de décès du conjoint
    famille_variables = ['ident_ind', 'cdatco', 'cdatdc', 'cdatse', 'couple']
    # CDATSE: Année de la séparation effective
    individus_variables = ['age', 'anais', 'etamatri', 'ident_ind', 'ident_instit', 'mnais', 'poids_hsi', 'sexe']
    institutions_variables = ['ident_instit', 'nvtypet']
    revenus_alloc_reconn_variables = ['ident_ind', 'rgir']
    # AGFINETU: Age de fin d'études initiales
    # DIP14: Diplôme le plus élevé (code regroupé)
    scolarite_variables = ['agfinetu', 'dip14', 'ident_ind']

    famille = survey.get_values(table = 'g_famille', variables = famille_variables)
    individus = survey.get_values(table = 'individus', variables = individus_variables)
    institutions = survey.get_values(table = 'info_instit', variables = institutions_variables)
    revenus_alloc_reconn = survey.get_values(
        table = 'l_revenus_alloc_reconn', variables = revenus_alloc_reconn_variables)
    scolarite = survey.get_values(table = 'j_scolarite', variables = scolarite_variables)

    dataframes = [famille, revenus_alloc_reconn, scolarite]

    result = individus
    for dataframe in dataframes:
        result = result.merge(dataframe, on = 'ident_ind')

    result = result.merge(institutions, on = 'ident_instit')

    list_variables_with_missing_values(result)

    return result


def impute_value(dataframe, endogeneous_variable, exogeneous_variables):
    assert 'age' in dataframe.columns
    rhs = None
    for exogeneous_variable in exogeneous_variables:
        rhs = rhs + " + " + exogeneous_variable if rhs is not None else exogeneous_variable
    formula = '{} ~ {}'.format(endogeneous_variable, rhs)

    cleared_dataframe = dataframe.copy()
    for variable in [endogeneous_variable] + exogeneous_variables:
        nans, missing_values, missing = compute_nans_and_missing_values(dataframe, variable)
        if nans != 0:
            cleared_dataframe = cleared_dataframe[cleared_dataframe[variable].notnull()].copy()
        if missing_values != 0:
            cleared_dataframe = cleared_dataframe[cleared_dataframe[variable] != missing].copy()

    y, X = patsy.dmatrices(formula, data = cleared_dataframe, return_type='dataframe')
    mod = sm.OLS(y, X)    # Describe model
    res = mod.fit()       # Fit model
    # print res.summary()   # Summarize model
    error_variance = res.scale  # the square root of `scale` is often called the standard error of the regression

    missing_dataframe = dataframe.copy()
    nans, missing_values, missing = compute_nans_and_missing_values(dataframe, endogeneous_variable)
    if missing_values != 0:
        missing_dataframe = missing_dataframe[missing_dataframe[endogeneous_variable] == missing].copy()

    for variable in exogeneous_variables:
        nans, missing_values, missing = compute_nans_and_missing_values(dataframe, variable)
        if nans != 0:
            missing_dataframe = missing_dataframe[missing_dataframe[variable].notnull()].copy()
        if missing_values != 0:
            missing_dataframe = missing_dataframe[missing_dataframe[variable] != missing].copy()

    y, X = patsy.dmatrices(formula, data = missing_dataframe, return_type='dataframe')

    predicted_dataframe = pandas.DataFrame(
        {endogeneous_variable: numpy.round(
            res.predict(X) + numpy.sqrt(error_variance) * numpy.random.randn(len(X.index))
            )},
        index = X.index)

    imputed_dataframe = dataframe.copy()
    imputed_dataframe.update(predicted_dataframe)
    return imputed_dataframe


def rename_variables(dataframe):
    result = dataframe.copy()
    result.sexe = result.sexe - 1

    new_by_old_name = dict(
        agfinetu = 'findet',
        etamatri = 'civilstate',
        poids_hsi = 'pond',
        )

    result['education_niveau'] = 9
    result.loc[result.dip14 // 10 == 7, 'education_niveau'] = 1
    result.loc[result.dip14 // 10 == 6, 'education_niveau'] = 2
    result.loc[result.dip14 // 10 == 5, 'education_niveau'] = 2  # BEP CAP
    result.loc[result.dip14 // 10 == 4, 'education_niveau'] = 3
    result.loc[result.dip14 // 10 == 3, 'education_niveau'] = 4
    result.loc[result.dip14 // 10 == 2, 'education_niveau'] = 5
    result.loc[result.dip14 // 10 == 1, 'education_niveau'] = 6

    # Ceux qui ne connaissent pas leur eta matrimonial sont considérés célibataires
    result.loc[result.etamatri == 9, 'etamatri'] = 1

    result.rename(columns = new_by_old_name, inplace = True)

    # TODO traiter ceux pour qui cdatse, cdatco = 9999 qui ne savent pas
    # Pour l'instant on met à -1

    result['duree_en_couple'] = 2009 - result['cdatco']
    result.loc[result.duree_en_couple.isnull(), 'duree_en_couple'] = -1

    result['duree_hors_couple'] = 2009 - result.cdatse
    result.loc[result.duree_hors_couple.isnull(), 'duree_hors_couple'] = -1

    assert (result.sexe.isin([0, 1])).all()
    assert (result.education_niveau.isin(range(1, 7) + [9])).all()
    assert (result.civilstate.isin(range(0, 5))).all()
    # assert (result.duree_en_couple >= -1).all()
    # assert (result.duree_hors_couple >= -1).all()
    return result


def expand_data(dataframe, weight_threshold):
    low_weight_dataframe = dataframe.query('pond < @weight_threshold')
    weights_of_random_picks = low_weight_dataframe.pond.sum()
    number_of_picks = int(weights_of_random_picks // weight_threshold)
    assert type(number_of_picks) == int
    log.info('Extracting {} from {} observations with weights lower than {} representing {} individuals'.format(
        number_of_picks,
        low_weight_dataframe.pond.count(),
        weight_threshold,
        weights_of_random_picks
        ))
    sample = low_weight_dataframe.sample(n = number_of_picks, weights = 'pond', random_state = 12345)
    sample.pond = weight_threshold
    return dataframe.query('pond >= @weight_threshold').append(sample)


def save(dataframe):
    config = Config()
    hsi_data_directory = config.get('raw_data', 'hsi_data_directory')
    if not os.path.exists(hsi_data_directory):
      os.mkdir(hsi_data_directory)
    dataframe.to_hdf(os.path.join(hsi_data_directory, "hsi_extract.h5"), 'individus_institutions')


def create_hsi_data():
    initial_dataframe = load_dataframe()
    exogeneous_variables = ['age', 'sexe']
    imputed_dataframe = initial_dataframe.copy()
    for endogeneous_variable in ['cdatco', 'cdatse', 'agfinetu', 'rgir']:
        compute_nans_and_missing_values(imputed_dataframe, endogeneous_variable)
        imputed_dataframe = impute_value(imputed_dataframe, endogeneous_variable, exogeneous_variables)
        nans, missing_values, missing = compute_nans_and_missing_values(imputed_dataframe, endogeneous_variable)

    dataframe = rename_variables(imputed_dataframe)
    dataframe = dataframe.dropna(subset = ['age']).query('age >= 60').copy()
    assert ((dataframe.anais >= 1900) & (dataframe.anais <= 2009)).all()
    assert dataframe.age.notnull().all()

    numpy.random.seed(12345)
    weight_threshold = 200
    final_dataframe = expand_data(dataframe, weight_threshold)
    final_dataframe.loc[final_dataframe.rgir <= 0, 'rgir'] = 0
    final_dataframe.loc[final_dataframe.rgir >= 6, 'rgir'] = 6
    save(final_dataframe)


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    create_hsi_data()
