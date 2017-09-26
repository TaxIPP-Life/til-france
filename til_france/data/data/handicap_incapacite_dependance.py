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
    modb['sexe'] = modb.sexe - 1  # 0: Homme, 1: Femme

    from gir_imputation import get_dataframe
    gir = get_dataframe()

    modb = modb.merge(gir, how = 'left')

    result = modb.query('age >= 60 ')[['age', 'sexe', 'dependance_entree_age', 'poidscor', 'gir']].copy()

    print result.groupby(['gir', 'sexe'])['poidscor'].sum() / 1e3

    result['dependance_duree'] = result.age - result.dependance_entree_age
    for sexe in result.sexe.unique():
        data = result.query('sexe == @sexe').copy()
        taux_dependants = (
            np.average((data.gir > 0) & (data.gir <= 4), weights = data.poidscor)
            )
        print sexe, taux_dependants

    clean = result.query(
        '(dependance_entree_age != 1000) & (dependance_entree_age != 999) & (gir > 0) & (gir <= 4)'
        ).copy()
    data = clean.dropna()[['dependance_duree', 'age', 'sexe', 'poidscor']].copy()
    assert data.notnull().all().all()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(data.age, data.dependance_duree)
    fig, ax = plt.subplots()
    data.dependance_duree.hist(bins = 30)
    # On regarde la probabilité selon l'âge d'avoir plus de 10 ans de dépendance
    import statsmodels.formula.api as smf
    data['dependance_duree_sup_10'] = (data.dependance_duree >= 10) * 1

    for sexe in data.sexe.unique():
        print sexe
        logit = smf.logit(formula='dependance_duree_sup_10 ~ age',
                          data=data.query('sexe == @sexe'))
        print logit.fit().summary()
        print (
            data.query('sexe == @sexe').groupby(
                ['dependance_duree_sup_10'])['poidscor'].sum() / data.query('sexe == @sexe').poidscor.sum()).round(2)

    for sexe in data.sexe.unique():
        print sexe
        data_duree = data.query('dependance_duree_sup_10 < 1 & sexe == @sexe').copy()
        fig, ax = plt.subplots()
        ax = data_duree.dependance_duree.hist(bins = 10, normed = True)
        ax.set_title('sexe = {}'.format(sexe))
        res = sm.Poisson(data_duree.dependance_duree, np.ones_like(data_duree.dependance_duree)).fit()
#        print res.summary()
#        ax = sns.distplot(data_duree.dependance_duree, kde = False, fit = stats.exponweib, rug = True)
#        ax.plot(range(10), scipy.stats.poisson.pmf(range(10), 1.4675))
#        See also: http://stackoverflow.com/questions/37500406/how-to-fit-a-poisson-distribution-with-seaborn/37500643#37500643

        import scipy.stats as stats
#        stats.exponweib.fit(data_duree.dependance_duree, floc=0)
        print (data_duree.groupby(['dependance_duree'])['poidscor'].sum() / data_duree.poidscor.sum()).round(2)

#        import scipy.stats as stats
#        stats.poisson fit(data_duree.dependance_duree, floc=0)


def save(dataframe):
    config = Config()


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    initial_dataframe = load_dataframe()
