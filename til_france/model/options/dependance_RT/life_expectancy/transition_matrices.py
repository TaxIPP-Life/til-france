#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import os
import pandas as pd
import patsy
import pkg_resources
import statsmodels.formula.api as smf


# Paths

til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )

assets_path = config_files_directory = os.path.join(
    til_france_path,
    'model',
    'options',
    'dependance_RT',
    'assets',
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
    filtered = (
        df[columns]
        .dropna()
        .rename(columns = {'scale5': 'initial_state'})
        )
    assert (filtered.isnull().sum() == 0).all()

    filtered["sexe"] = filtered["sexe"].astype('int').astype('category')
    filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')
    # filtered.initial_state.value_counts()
    return filtered


def build_estimation_sample(initial_state, final_states, sex = None):
    assert sex in ['male', 'female']
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
    if sex:
        if sex == 'male':
            sample = sample.query('sexe == 1').copy()
        elif sex == 'female':
            sample = sample.query('sexe == 2').copy()
    sample["final_state"] = sample["final_state"].astype('int').astype('category')
    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states)

    return sample.reset_index()


def build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = None):
    assert sex in ['male', 'female', 'all']
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
            .assign(sex = sex)
            [['sex', 'age', 'initial_state', 'final_state', 'probability']]
            )
    return pd.concat(
        transition_matrices,
        ignore_index = True
        ).set_index(
            ['sex', 'age', 'initial_state', 'final_state']
            )


def estimate_model(initial_state, final_states, formula, sex = None, variables = ['age', 'final_state']):
    assert sex in ['male', 'female']
    sample = build_estimation_sample(initial_state, final_states, sex = sex)
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


def direct_compute_predicition(initial_state, final_states, formula, formatted_params, sex = None):
    assert sex in ['male', 'female']
    computed_prediction = build_estimation_sample(initial_state, final_states, sex = sex)
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


def compute_prediction(initial_state, final_states, formula = None, variables = ['age'], exog = None, sex = None):
    assert sex in ['male', 'female']
    sample = build_estimation_sample(initial_state, final_states, sex = sex)
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


def get_transitions_from_formula(formula = None):
    transitions = None
    for sex in ['male', 'female']:
        assert formula is not None
        proba_by_initial_state = dict()
        exog = pd.DataFrame(dict(age = range(65, 120)))
        for initial_state, final_states in final_states_by_initial_state.iteritems():
            proba_by_initial_state[initial_state] = pd.concat(
                [
                    exog,
                    compute_prediction(initial_state, final_states, formula, exog = exog, sex = sex)
                    ],
                axis = 1,
                )
        transitions = pd.concat([
            transitions,
            build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = sex)
            ])

    transitions.reset_index().set_index('sex', 'age', 'initial_state', 'fianl_state')
    return transitions


def test(formula = None,
     initial_state = 0,
     final_states = [0, 1, 4, 5],
     sex = None):
    assert formula is not None
    assert sex is not None
    result, formatted_params = estimate_model(initial_state, final_states, formula, sex = sex)
    print(result.summary(alpha = .1))
    print(formatted_params)
    computed_prediction = direct_compute_predicition(
        initial_state, final_states, formula, formatted_params, sex = sex)
    prediction = compute_prediction(initial_state, final_states, formula, sex = sex)
    diff = computed_prediction[prediction.columns] - prediction
    assert (diff.abs().max() < 1e-10).all(), "error is too big: {} > 1e-10".format(diff.abs().max())


if __name__ == '__main__':
    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'
    test(formula = formula, sex = 'female')
