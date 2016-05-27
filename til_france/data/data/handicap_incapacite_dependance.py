# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import pandas as pd
import patsy
import pkg_resources
import seaborn as sns
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
    np.random.seed(12345)
    # load data
    survey_collection = SurveyCollection.load(
        collection = 'hid', config_files_directory = config_files_directory)
    survey = survey_collection.get_survey('hid_domicile_1999')

    modb = survey.get_values(table = 'modb_c')
    ages = [column for column in modb.columns if column.endswith('a')]
    modb = modb[['ident', 'numind'] + ages].copy()
    modb.fillna(1000, inplace = True)
    modb['dependance_entree_age'] = modb[ages].min(axis = 1)

    mindiv = survey.get_values(table = 'mindiv_c', variables = ['ident', 'numind', 'age', 'sexe', 'poidscor'])

    modb = modb.merge(mindiv, how = 'left')
    assert 'age' in modb.columns
    assert 'sexe' in modb.columns
    modb['sexe'] = modb.sexe - 1
    result = modb.query('age >= 60')[['age', 'sexe', 'dependance_entree_age', 'poidscor']].copy()
    result['dependance_duree'] = result.age - result.dependance_entree_age

    from gir_imputation import get_dataframe
    gir = get_dataframe()


    for sexe in result.sexe.unique():
        data = result.query('sexe == @sexe').copy()
        taux_dependants = (
            np.average(data.dependance_entree_age != 1000, weights = data.poidscor)
            )
        print sexe, taux_dependants

    clean = result.query('(dependance_entree_age != 1000) & (dependance_entree_age != 999)').copy()
    data = clean.dropna()[['dependance_duree', 'age', 'sexe']].copy()
    assert data.notnull().all().all()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(data.age, data.dependance_duree)
    data.dependance_duree.hist(bins = 30)
    # On regarde la probabilité selon l'âge d'avoir plus de 10 ans de dépendance
    import statsmodels.formula.api as smf
    data['dependance_duree_sup_10'] = (data.dependance_duree >= 10) * 1

    for sexe in data.sexe.unique():
        print sexe
        logit = smf.logit(formula='dependance_duree_sup_10 ~ age',
                          data=data.query('sexe == @sexe'))
        logit.fit().summary()

    for sexe in data.sexe.unique():
        print sexe
        data_duree = data.query('dependance_duree_sup_10 < 1 & sexe == @sexe').copy()
        data_duree.dependance_duree.hist(bins = 10)

        import scipy.stats
        res = sm.Poisson(data_duree.dependance_duree, np.ones_like(data_duree.dependance_duree)).fit()
        res.summary()

        import scipy.stats as stats
        stats.exponweib.fit(data_duree.dependance_duree, floc=0)

        import scipy.stats as stats
        stats.poisson.fit(data_duree.dependance_duree, floc=0)

    formula = 'dependance_duree ~ age'
    import statsmodels.formula.api as smf
    glm_exponential = smf.glm(formula, data, family = sm.families.family.Gamma())
    exponential_result = glm_exponential.fit()
    print exponential_result.summary()
    data.age
    fig2, ax2 = plt.subplots()
#
    yhat = exponential_result.predict(data)

    ax2.scatter(yhat, data.dependance_duree)

    y, X = patsy.dmatrices(formula, data = data, return_type='dataframe')
    ols = sm.OLS(y, X)    # Describe model
    # Fit model
    print ols.fit().summary()

    ax.scatter(ols.predict(X), data.age)

    result.replace({'dependance_entree_age': {999: np.nan}}, inplace = True)

    interp = result.query('dependance_entree_age != 1000').copy()

    assert interp.isnull().any().any()
    interp['dependance_entree_age'] = interp.groupby(['sexe', 'age'])['dependance_entree_age'].transform(
        lambda x: x.fillna(x.mean()))
    assert interp.notnull().all().all()

    result = result.combine_first(interp)
    assert result.notnull().all().all()

    result.replace({'dependance_entree_age': {1000: np.nan}}, inplace = True)
    assert result.isnull().any().any()

#    dependance_entree_age = pd.DataFrame()
#    for sexe in result.sexe.unique():
#        df = result.query('sexe == @sexe').copy()
#        data = df.dependance_entree_age.dropna().values
#        sns.distplot(data)
#
#        weights = df.poidscor.loc[df.dependance_entree_age.notnull()].values
#        density = sm.nonparametric.KDEUnivariate(
#            data,
#            )
#        density.fit(
#            weights = weights,
#            fft = False,
#            )
#        import matplotlib.pyplot as plt
#        plt.plot([density.evaluate(x) for x in range(0,120)])
#        plt.show()
#        cdf = density.cdf
#        _min, _max = int(np.min(data)), int(np.max(data))
#        # x =  _min + i * (_max - _min) / (len(cdf) - 1)
#        i_s = [
#            np.round((x -  _min) * (len(cdf) - 1) / (_max - _min))
#            for x in range(_min, _max)
#            ]
#        density_data_frame = pd.DataFrame({
#            'density': [cdf[i+1] - cdf[i] for i in i_s],
#            'age': range(_min, _max)
#            })
#        density_data_frame.plot(y = 'density', x ='age')
#
#        density_data_frame['sexe'] = sexe
#        print sexe
#        dependance_entree_age = pd.concat([dependance_entree_age, density_data_frame], axis = 0)

    # formula = 'dependance_entree_age ~ age'

#    for sexe in result.sexe.unique():
        len(yhat)

        TODO use an exponential distribution

def save(dataframe):
    config = Config()
    hsi_data_directory = config.get('raw_data', 'hsi_data_directory')
    dataframe.to_hdf(os.path.join(hsi_data_directory, "hsi_extract.h5"), 'individus_institutions')


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    initial_dataframe = load_dataframe()
