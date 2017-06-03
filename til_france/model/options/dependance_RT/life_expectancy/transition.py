#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import numpy as np
import os
import pandas as pd
import patsy
import pkg_resources
import statsmodels.formula.api as smf


# Paths
assets_path = config_files_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'assets',
    )

life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )

paquid_path = u'/home/benjello/data/dependance/paquid_panel_3.csv'
# paquid_dta_path = u'/home/benjello/data/dependance/paquid_panel_3_mahdi.dta'
# df2 = pd.read_stata(paquid_dta_path, encoding = 'utf-8')


# Transition matrix structure
final_states_by_initial_state = {
    0: [0, 1, 4, 5],
    1: [0, 1, 2, 4, 5],
    2: [1, 2, 3, 4, 5],
    3: [2, 3, 4, 5],
    4: [4, 5],
    }


def get_filtered_paquid_data():
    df = pd.read_csv(paquid_path)
    columns = ['numero', 'annee', 'age', 'scale5', 'sexe']
    filtered = (df[columns]
        .dropna()
        .rename(columns = {'scale5': 'initial_state'})
        )
    assert (filtered.isnull().sum() == 0).all()

    filtered["sexe"] = filtered["sexe"].astype('int').astype('category')
    filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')
    # filtered.initial_state.value_counts()
    return filtered


def build_estimation_sample(initial_state, final_states, sexe = None):
    assert sexe is not None
    filtered = get_filtered_paquid_data()
    assert initial_state in final_states
    filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)
    sample = (filtered
        .query('(initial_state == {}) & (final_state in {})'.format(
            initial_state,
            final_states,
            ))
        .dropna()
        )
    if sexe:
        assert sexe in ['male', 'homme', 'female', 'femme']
        if sexe == 'male' or sexe == 'homme':
            sample = sample.query('sexe == 1').copy()
        elif sexe == 'female' or sexe == 'femme':
            sample = sample.query('sexe == 2').copy()
    sample["final_state"] = sample["final_state"].astype('int').astype('category')
    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states)

    return sample.reset_index()


def estimate_model(initial_state, final_states, formula, sexe = None, variables = ['age', 'final_state']):
    assert sexe is not None
    sample = build_estimation_sample(initial_state, final_states, sexe = sexe)
    result = smf.mnlogit(
        formula = formula,
        data = sample[variables],
        ).fit()

    formatted_params = result.params.copy()
    formatted_params.columns = sorted(set(final_states) - set([initial_state]))

    def rename_index_func(index):
        index = index.lower()
        if index.startswith('i('):
            index = index[1:]
        elif index.startswith('intercept'):
            index = '1'
        return index

    formatted_params.rename(index = rename_index_func, inplace = True)
    formatted_params[initial_state] = 0
    return result, formatted_params


def direct_compute_predicition(initial_state, final_states, formula, formatted_params, sexe = None):
    assert sexe is not None
    computed_prediction = build_estimation_sample(initial_state, final_states, sexe = sexe)
    for final_state, column in formatted_params.iteritems():
        proba_str = "exp({})".format(
            " + ".join([index + " * " + str(value) for index, value in zip(column.index, column.values)])
            )
        computed_prediction['proba_etat_{}'.format(final_state)] = computed_prediction.eval(proba_str)

    computed_prediction['z'] = computed_prediction[[
        col for col in computed_prediction.columns if col.startswith('proba')
        ]].sum(axis = 1)

    for col in computed_prediction.columns:
        if col.startswith('proba'):
            computed_prediction[col] = computed_prediction[col] / computed_prediction['z']

    return computed_prediction


def compute_prediction(initial_state, final_states, formula = None, variables = ['age'], exog = None, sexe = None):
    assert sexe is not None
    sample = build_estimation_sample(initial_state, final_states, sexe = sexe)
    if exog is None:
        exog = sample[variables]

    result = smf.mnlogit(
        formula = formula,
        data = sample,
        ).fit()
    expurged_formula = formula.split('~', 1)[-1]
    x = patsy.dmatrix(expurged_formula, data= exog)  # df is data for prediction
    prediction = result.predict(x, transform=False)
    (abs(prediction.sum(axis = 1) - 1) < .00001).all()
    prediction = pd.DataFrame(prediction)
    prediction.columns = ['proba_etat_{}'.format(state) for state in sorted(final_states)]
    return prediction.reset_index(drop = True)


def get_proba_by_initial_state(formula = None, sexe = None):
    assert sexe is not None
    assert formula is not None

    proba_by_initial_state = dict()
    exog = pd.DataFrame(dict(age = range(65, 120)))
    for initial_state, final_states in final_states_by_initial_state.iteritems():
        proba_by_initial_state[initial_state] = pd.concat(
            [
                exog,
                compute_prediction(initial_state, final_states, formula, exog = exog, sexe = sexe)
                ],
            axis = 1,
            )
    return proba_by_initial_state


def get_predicted_mortality_table(formula = None, sexe = None):
    assert sexe is not None
    assert formula is not None
    proba_by_initial_state = get_proba_by_initial_state(formula = formula, sexe = sexe)
    exog = pd.DataFrame(dict(age = range(65, 120)))
    mortality_by_initial_state = exog
    for initial_state in final_states_by_initial_state.keys():
        mortality_by_initial_state = pd.concat(
            [
                mortality_by_initial_state,
                pd.DataFrame({
                    initial_state: (1 - np.sqrt(1 - proba_by_initial_state[initial_state]['proba_etat_5'])),
                    }),
                ],
            axis = 1,
            )
    mortality_table = pd.melt(
        mortality_by_initial_state,
        id_vars=['age'],
        value_vars=[0, 1, 2, 3, 4],
        var_name = 'initial_state',
        value_name = 'mortality',
        )
    return mortality_table


def test(formula = None,
     initial_state = 0,
     final_states = [0, 1, 4, 5],
     sexe = None):
    assert formula is not None
    assert sexe is not None
    result, formatted_params = estimate_model(initial_state, final_states, formula, sexe = sexe)
    print(result.summary(alpha = .1))
    print(formatted_params)
    computed_prediction = direct_compute_predicition(
        initial_state, final_states, formula, formatted_params, sexe = sexe)
    prediction = compute_prediction(initial_state, final_states, formula, sexe = sexe)
    diff = computed_prediction[prediction.columns] - prediction
    assert (diff.abs().max() < 1e-10).all(), "error is too big: {} > 1e-10".format(diff.abs().max())


def get_historical_mortalite_by_sex():
    mortalite_by_sex = dict()
    mortalite_by_sex['male'] = (pd.read_excel(life_table_path, sheetname = 'france-male')[['Year', 'Age', 'qx']]
        .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
        )
    mortalite_by_sex['female'] = (pd.read_excel(life_table_path, sheetname = 'france-female')[['Year', 'Age', 'qx']]
        .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
        )
    return mortalite_by_sex


def plot_paquid_comparison(formula = None, sexe = None):
    assert formula is not None
    mortality_table = get_predicted_mortality_table(sexe = sexe)
    filtered = get_filtered_paquid_data()
    filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)
    filtered['age_group_10'] = 10 * (filtered.age / 10).apply(np.floor).astype('int')
    filtered['age_group_5'] = 5 * ((filtered.age) / 5).apply(np.floor).astype('int')
    if sexe:
        if sexe in ['homme', 'male']:
            sexe_nbr = 1
        elif sexe in ['femme', 'female']:
            sexe_nbr = 2

        filtered = filtered.query('sexe == {}'.format(sexe_nbr))

    test_df = filtered.query('(annee == 2003) & (initial_state != 5)').dropna()
    test_df.initial_state.value_counts()

    test_df.groupby(['age_group_5'])[['final_state']].apply(
        lambda x: 1 - np.sqrt(1 - 1.0 * (x.final_state == 5).sum() / x.count())
        )

    from til_france.targets.population import build_mortality_rates
    mortalite_insee = build_mortality_rates()[sexe][2007]
    mortalite_insee.name = 'mortalite_insee'

    mortalite_by_sex = get_historical_mortalite_by_sex()

    profile = filtered.query('initial_state != 5').dropna()
    profile.age.round().value_counts()

    profile['age'] = profile.age.round()

    sample_mortality_profile = (profile
        .merge(mortality_table, on =['age', 'initial_state'], how = 'left')
        .groupby(['age'])['mortality']
        .mean()
        )

    mortalite_1988 = (mortalite_by_sex[sexe]
        .query('annee == 1988')
        .reset_index()
        .loc[:109, ['age', 'mortalite']]
        .astype(dict(age = 'int'))
        .set_index('age')
        .rename(columns = dict(mortalite = 'mortalite_1988'))
        )

    plot_data = pd.concat([sample_mortality_profile, mortalite_insee, mortalite_1988], axis = 1)
    plot_data.index.name = 'age'
    ax = plot_data.query('age > 60').plot()
    ax.get_figure().savefig('mortalite.png')

    plot_data['ratio'] = plot_data.eval('mortality / mortalite_insee')

    ax2 = plot_data.query('age > 60').plot(y = 'ratio')
    ax2.get_figure().savefig('ratio_mortalite.png')

    sexe_nbr = 0 if sexe == 'male' else 1

    df = (pd.read_csv(os.path.join(assets_path, 'dependance_niveau.csv'), index_col = 0)
        .query('(period == 2010) and (age >= 60)'))

    assert (df.query('dependance_niveau == -1')['total'] == 0).all()

    mortalite_after_imputation = (df
        .query('dependance_niveau != -1')
        .query('sexe == @sexe_nbr')
        .query('dependance_niveau != 5')
        .rename(columns = dict(dependance_niveau = 'initial_state'))
        .merge(mortality_table, how = 'inner')
        .groupby(['age'])[['total', 'mortality']].apply(lambda x: (x.total * x.mortality).sum() / x.total.sum())
        )
    mortalite_after_imputation.name = 'mortalite_after_imputation'

    plot_data = pd.concat(
        [sample_mortality_profile, mortalite_insee, mortalite_1988, mortalite_after_imputation],
        axis = 1,
        )
    plot_data.index.name = 'age'
    ax = plot_data.query('age > 60').plot()
    ax.get_figure().savefig('mortalite.png')


def get_calibration(period = None, formula = None, sexe = None):
    assert period is not None
    assert formula is not None
    assert sexe in ['homme', 'male', 'femme', 'female']
    mortality_table = get_predicted_mortality_table(formula = formula, sexe = sexe)
    if sexe in ['homme', 'male']:
        sexe_nbr = 1
    elif sexe in ['femme', 'female']:
        sexe_nbr = 2

    mortalite_after_imputation = (pd.read_csv(os.path.join(assets_path, 'dependance_niveau.csv'), index_col = 0)
        .query('(period == @period) and (age >= 60)')
        .query('dependance_niveau != -1')
        .query('sexe == @sexe_nbr')
        .query('dependance_niveau != 5')
        .merge(
            mortality_table.rename(columns = dict(initial_state = 'dependance_niveau')),
            how = 'inner')
        )

    global_mortalite_after_imputation = (mortalite_after_imputation
        .groupby(['age'])[['total', 'mortality']].apply(lambda x: (x.total * x.mortality).sum() / x.total.sum())
        .reset_index()
        .rename(columns = {0: 'avg_mortality'})
        )

    mortalite_by_sex = get_historical_mortalite_by_sex()
    mortalite_reelle = (mortalite_by_sex[sexe]
        .query('annee == @period')
        .reset_index()
        .loc[:109, ['age', 'mortalite']]
        .astype(dict(age = 'int'))
        .rename(columns = dict(annee = 'period'))
        )

    mortalite_reelle['sexe'] = sexe_nbr

    model_to_target = (global_mortalite_after_imputation.merge(mortalite_reelle)
        .eval('cale_mortality_1_year = mortalite / avg_mortality', inplace = False)
        .eval('mortalite_2_year = 1 - (1 - mortalite) ** 2', inplace = False)
        .eval('avg_mortality_2_year = 1 - (1 - avg_mortality) ** 2', inplace = False)
        .eval('cale_mortality_2_year = mortalite_2_year / avg_mortality_2_year', inplace = False)
        )
    return model_to_target


def get_transtions(formula = None, sexe = None):
    proba_by_initial_state = get_proba_by_initial_state(formula = formula, sexe = sexe)
    transition_matrices = list()
    for initial_state, proba_dataframe in proba_by_initial_state.iteritems():
        transition_matrices.append(
            pd.melt(
                proba_dataframe,
                id_vars = ['age'],
                var_name = 'final_state',
                value_name = 'probability',
                )
            .replace({'final_state': dict([('proba_etat_{}'.format(index), index) for index in range(6)])})
            .assign(initial_state = initial_state)
            [['age', 'initial_state', 'final_state', 'probability']]
            )

    return pd.concat(transition_matrices, ignore_index = True).set_index(['age', 'initial_state', 'final_state'])


def get_calibrated_transition(formula = None, period = None, sexe = None):
    assert period is not None
    transitions = get_transtions(formula = formula, sexe = sexe)
    calibration = get_calibration(formula = formula, period = period, sexe = sexe)
    mortality = (transitions
        .reset_index()
        .query('final_state == 5')
        .merge(calibration[['age', 'cale_mortality_2_year']], on = 'age')
        .eval('calibrated_probability = probability * cale_mortality_2_year', inplace = False)
        .eval(
            'cale_other_transitions = (1 - probability * cale_mortality_2_year) / (1 - probability)',
            inplace = False,
            )
        )

    cale_other_transitions = mortality[['age', 'initial_state', 'cale_other_transitions']].copy()
    other_transitions = (transitions
        .reset_index()
        .query('final_state != 5')
        .merge(cale_other_transitions)
        .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
        )

    calibrated_transitions = (pd.concat([mortality, other_transitions])
        .set_index(['age', 'initial_state', 'final_state'])
        .sort_index()
        )
    diff = (
        calibrated_transitions.reset_index().groupby(['age', 'initial_state'])['calibrated_probability'].sum() - 1)
    assert (diff.abs().max() < 1e-10).all(), "error is too big: {} > 1e-10".format(diff.abs().max())
    calibrated_transitions['calibrated_probability'].reset_index()
    return calibrated_transitions


if __name__ == '__main__':
    # formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2)'
    # formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3) + I(((age - 80))**4)'
    # formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1)**2) + I(((age - 80) * 0.1)**3) + I(((age - 80) * 0.1)**4) + I(((age - 80) * 0.1)**5)'

    # test(formula = formula, sexe = 'female')
    # plot_paquid_comparison(formula = formula, sexe = 'female')

    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'
    period = 2010
    sexe = 'male'
    filename = os.path.join('/home/benjello/data/til/input/test.csv')

    def export_calibrated_transitions_to_liam(formula = None, period = None, sexe = None, filename = None):
        calibrated_transitions = get_calibrated_transition(period = period, formula = formula, sexe = sexe)
        age_max = calibrated_transitions.index.levels[0].max()
        null_fill_by_year = calibrated_transitions['calibrated_probability'].reset_index().query('age == 65').copy()
        null_fill_by_year['calibrated_probability'] = 0
        pre_65_null_fill = pd.concat([
            null_fill_by_year.assign(age = i).copy()
            for i in range(0, 65)
            ]).reset_index(drop = True)
        elder_null_fill = pd.concat([
            null_fill_by_year.assign(age = i).copy()
            for i in range(age_max + 1, 121)
            ]).reset_index(drop = True)

        age_full = pd.concat([
            pre_65_null_fill,
            calibrated_transitions['calibrated_probability'].reset_index(),
            elder_null_fill
            ]).reset_index(drop = True)

        age_full['period'] = period
        result = age_full[
            ['period', 'age', 'initial_state', 'final_state', 'calibrated_probability']
            ].set_index(['period', 'age', 'initial_state', 'final_state'])
        if filename is not None:
            result.unstack().fillna(0).to_csv(filename, header = False)
        return age_full[['period', 'age', 'initial_state', 'final_state']]


        verbatim_header = '''period,age,initial_state,final_state,,,,,
,,,0,1,2,3,4,5'''


    bim
    x = (calibrated_transitions
        .query('initial_state == 0')['calibrated_probability']
        .reset_index()
        .drop('initial_state', axis = 1)
        .set_index(['age', 'final_state'])
        .unstack()
        )

    x
