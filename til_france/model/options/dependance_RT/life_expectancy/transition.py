#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels as sm
sm.api;ad
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

# Build final_state
filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)


state = (filtered
    .query('(initial_state == 0) & (final_state in [0, 1, 4, 5])')
    .dropna()
    )

state["final_state"] = state["final_state"].astype('int').astype('category')
state.final_state.value_counts()

result = smf.mnlogit(
    formula = 'final_state ~ I(age - 80) + I((age - 80)**2) + I((age - 80)**3)',
    data = state[['age', 'final_state']],
    ).fit()

result = smf.mnlogit(
    formula = 'final_state ~ age ',
    data = state[['age', 'final_state']],
    ).fit()

params = result.params
print result.summary()


prediciton = result.predict()

import patsy
exog = state[['age']].query('age > 80')
exog = state[['age']]
x = patsy.dmatrix("I(age - 80) + I((age - 80)**2) + I((age - 80)**3)", data= exog)  # df is data for prediction
test2 = result.predict(x, transform=False)

prediciton = result.predict(])


(abs(prediciton.sum(axis = 1) - 1) < .00001).all()


computed_prediction = state.copy()
computed_prediction['proba_0'] = 1  # exp(0)
computed_prediction['proba_1'] = computed_prediction.eval('')