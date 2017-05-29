#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels as sm

import statsmodels.formula.api as smf


paquid_path = u'/home/benjello/data/dependance/paquid_panel_3.csv'

#paquid_dta_path = u'/home/benjello/data/dependance/paquid_panel_3_mahdi.dta'
#df2 = pd.read_stata(paquid_dta_path, encoding = 'utf-8')

df = pd.read_csv(paquid_path)

columns = ['numero', 'annee', 'age', 'scale5', 'sexe']
filtered = (df[columns]
    .dropna()
    .rename(columns = {'scale5': 'initial_state'})
    )
assert (filtered.isnull().sum() == 0).all()

filtered["sexe"] = filtered["sexe"].astype('int').astype('category')
filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')
filtered.initial_state.value_counts()



def build_estimation_sample(initial_state, final_states):
    assert initial_state in final_states
    filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)
    sample = (filtered
        .query('(initial_state == {}) & (final_state in {})'.format(
            initial_state,
            final_states,
            ))
        .dropna()
        )
    sample["final_state"] = sample["final_state"].astype('int').astype('category')
    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states)

    return sample.reset_index()


def estimate_model(initial_state, final_states, formula, varaibles = ['age', 'final_state', 'sexe']):
    sample = build_estimation_sample(initial_state, final_states)
    result = smf.mnlogit(
        formula = formula,
        data = sample[varaibles],
        ).fit()
    print result.params
    print result.summary()

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


def direct_compute_predicition(initial_state, final_states, formula, formatted_params):
    computed_prediction = build_estimation_sample(initial_state, final_states)
    for final_state, column in formatted_params.iteritems():
        print final_state
        print column
        proba_str = "exp({})".format(
            " + ".join([index + " * " + str(value) for index, value in zip(column.index, column.values)])
            )
        print proba_str
        computed_prediction['proba_{}'.format(final_state)] = computed_prediction.eval(proba_str)

    computed_prediction['z'] =  computed_prediction[[
        col for col in computed_prediction.columns if col.startswith('proba')
        ]].sum(axis = 1)

    for col in computed_prediction.columns:
        if col.startswith('proba'):
            computed_prediction[col] = computed_prediction[col] / computed_prediction['z']

    return computed_prediction


def compute_prediction(initial_state, final_states, formula, result, exog = None):
    import patsy
    if exog is None:
        exog = build_estimation_sample(initial_state, final_states)[['age', 'sexe']]
    x = patsy.dmatrix(formula.split('~', 1)[-1], data= exog)  # df is data for prediction
    prediction = result.predict(x, transform=False)
    (abs(prediction.sum(axis = 1) - 1) < .00001).all()
    prediction = pd.DataFrame(prediction)
    prediction.columns = ['proba_{}'.format(state) for state in sorted(final_states)]
    return prediction


initial_state = 0
final_states = [0, 1, 4, 5]
formula = 'final_state ~ I(age - 80) + I((age - 80)**2) + I((age - 80)**3)'

result, formatted_params = estimate_model(initial_state, final_states, formula)

print formatted_params

computed_prediction = direct_compute_predicition(initial_state, final_states, formula, formatted_params)
prediction = compute_prediction(initial_state, final_states, formula, result)
diff = computed_prediction[prediction.columns] - prediction

print diff.min()
print diff.max()


transition_by_age = pd.DataFrame(dict(age = range(65, 120)))
probas_by_age = compute_prediction(initial_state, final_states, formula, result, exog = transition_by_age)
transition_by_age = transition_by_age.merge(probas_by_age, left_index = True, right_index = True)
