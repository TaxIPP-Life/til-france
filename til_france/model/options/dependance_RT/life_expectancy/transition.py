#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import numpy as np
import os
import pandas as pd
import patsy
import pkg_resources
import statsmodels.formula.api as smf


paquid_path = u'/home/benjello/data/dependance/paquid_panel_3.csv'

# paquid_dta_path = u'/home/benjello/data/dependance/paquid_panel_3_mahdi.dta'
# df2 = pd.read_stata(paquid_dta_path, encoding = 'utf-8')

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


def build_estimation_sample(initial_state, final_states, sexe = None):
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
            sample = sample.query('sexe == 1')
        elif sexe == 'female' or sexe == 'femme':
            sample = sample.query('sexe == 2')
    sample["final_state"] = sample["final_state"].astype('int').astype('category')
    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states)

    return sample.reset_index()


def estimate_model(initial_state, final_states, formula, sexe = None, variables = ['age', 'final_state']):
    sample = build_estimation_sample(initial_state, final_states, sexe = sexe)

    result = smf.mnlogit(
        formula = formula,
        data = sample[variables],
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


def direct_compute_predicition(initial_state, final_states, formula, formatted_params, sexe = None):
    computed_prediction = build_estimation_sample(initial_state, final_states, sexe = sexe)
    for final_state, column in formatted_params.iteritems():
        proba_str = "exp({})".format(
            " + ".join([index + " * " + str(value) for index, value in zip(column.index, column.values)])
            )
        print proba_str
        computed_prediction['proba_etat_{}'.format(final_state)] = computed_prediction.eval(proba_str)

    computed_prediction['z'] = computed_prediction[[
        col for col in computed_prediction.columns if col.startswith('proba')
        ]].sum(axis = 1)

    for col in computed_prediction.columns:
        if col.startswith('proba'):
            computed_prediction[col] = computed_prediction[col] / computed_prediction['z']

    return computed_prediction


def compute_prediction(initial_state, final_states, formula, variables = ['age'], exog = None, sexe = None):
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


def test():
    initial_state = 0
    final_states = [0, 1, 4, 5]
    sexe = 'homme'
    formula = 'final_state ~ I(age - 80) + I((age - 80)**2) + I((age - 80)**3)'

    result, formatted_params = estimate_model(initial_state, final_states, formula, sexe = sexe)

    print formatted_params

    computed_prediction = direct_compute_predicition(initial_state, final_states, formula, formatted_params, sexe = sexe)
    prediction = compute_prediction(initial_state, final_states, formula)
    diff = computed_prediction[prediction.columns] - prediction

    print diff.min()
    print diff.max()


# test()

formula = 'final_state ~ I(age - 80) + I((age - 80)**2) + I((age - 80)**3)'
sexe = 'male'
final_states_by_initial_state = {
    0: [0, 1, 4, 5],
    1: [0, 1, 2, 4, 5],
    2: [1, 2, 3, 4, 5],
    3: [2, 3, 4, 5],
    4: [4, 5],
    }


proba_by_initial_state = dict()
exog = pd.DataFrame(dict(age = range(65, 120)))
mortality_by_initial_state = exog

for initial_state, final_states in final_states_by_initial_state.iteritems():
    proba_by_initial_state[initial_state] = pd.concat(
        [
            exog,
            compute_prediction(initial_state, final_states, formula, exog = exog, sexe = sexe)
            ],
        axis = 1,
        )
    mortality_by_initial_state = pd.concat(
        [
            mortality_by_initial_state,
            pd.DataFrame({
                initial_state: (1 - np.sqrt(1 - proba_by_initial_state[initial_state]['proba_etat_5'])),
                }),
            ],
        axis = 1,
        )


filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)
filtered['age_group_10'] = 10 * (filtered.age / 10).apply(np.floor).astype('int')
filtered['age_group_5'] = 5 * ((filtered.age) / 5).apply(np.floor).astype('int')
if sexe:
    if sexe in ['homme', 'male']:
        sexe_nbr = 1
    elif sexe in ['femme', 'female']:
        sexe_nbr = 2

    filtered = filtered.query('sexe == {}'.format(sexe_nbr))


test = filtered.query('(annee == 2003) & (initial_state != 5)').dropna()
test.initial_state.value_counts()

test.groupby(['age_group_5'])[['final_state']].apply(
    lambda x: 1 - np.sqrt(1 - 1.0 * (x.final_state == 5).sum() / x.count())
    )


from til_france.targets.population import build_mortality_rates
mortalite_insee = build_mortality_rates()[sexe][2007]
mortalite_insee.name = 'mortalite_insee'

mortalite_by_sex = dict()
mortalite_by_sex['male'] = (pd.read_excel(life_table_path, sheetname = 'france-male')[['Year', 'Age', 'qx']]
    .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
    )
mortalite_by_sex['female'] = (pd.read_excel(life_table_path, sheetname = 'france-female')[['Year', 'Age', 'qx']]
    .rename(columns = dict(Year = 'annee', Age = 'age', qx = 'mortalite'))
    )

profile = filtered.query('initial_state != 5').dropna()
profile.age.round().value_counts()

profile['age'] = profile.age.round()

mortality_table = pd.melt(
    mortality_by_initial_state,
    id_vars=['age'],
    value_vars=[0, 1, 2, 3, 4],
    var_name = 'initial_state',
    value_name = 'mortality',
    )


df = profile.merge(mortality_table, on =['age', 'initial_state'], how = 'left')
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

ax2 = plot_data.query('age').plot(y = 'ratio')
ax2.get_figure().savefig('ratio_mortalite.png')
