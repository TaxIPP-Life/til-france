#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import pandas as pd
import statsmodels.formula.api as smf


paquid_path = u'/home/benjello/data/dependance/paquid_panel_3.csv'

#paquid_dta_path = u'/home/benjello/data/dependance/paquid_panel_3_mahdi.dta'
#df2 = pd.read_stata(paquid_dta_path, encoding = 'utf-8')

df = pd.read_csv(paquid_path)

for col in df.columns:
    print col


columns = ['numero', 'annee', 'age', 'scale5', 'sexe']
filtered = (df[columns]
    .dropna()
    .rename(columns = {'scale5': 'initial_state'})
    )
assert (filtered.isnull().sum() == 0).all()

filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')
filtered.initial_state.value_counts()

# Build final_state
filtered['final_state'] = filtered.groupby('numero')['initial_state'].shift(-1)


state = (filtered
    .query('initial_state == 0')
    .dropna()
    )

result = smf.mnlogit(
    formula = 'final_state ~ C(sexe) + (age - 80)',
    data = state,
    ).fit()

result.params
print result.summary()
