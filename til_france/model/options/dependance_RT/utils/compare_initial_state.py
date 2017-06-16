#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import os
import pandas as pd


from til_france.model.options.dependance_RT.life_expectancy.transition import assets_path

# Compare initial_states

df = pd.read_csv('/home/benjello/data/til/output/dependance_RT_paquid/dependance_niveau2.csv').iloc[:-1]
df['period'] = df['period'].astype(float).astype(int)

    .query('(period == 2010.0) and (age >= 65)'))

plot_data = df.groupby(['sexe', 'age', 'dependance_niveau'])['total'].sum().unstack().drop([-1, 5], axis = 1)

plot_data['total'] = plot_data.sum(axis = 1).squeeze()
for col in plot_data:
    plot_data[col] = plot_data[col] / plot_data.total

plot_data.query('sexe == 0').plot()
plot_data.query('sexe == 1').plot()


for sexe in ['homme', 'femme']:
    hsm_data = (pd.read_excel(os.path.join(assets_path, 'Tableau_{}s.xlsx'.format(sexe)))
        .drop('sexe', axis = 1)
        .fillna(method = 'backfill')
        .rename(columns = {u"âge": 'age'})
        .set_index(['age' ,'scale'])
        .unstack()
        .iloc[1:]
        .squeeze()
        )


    hsm_data = hsm_data.fillna(0)['effectif'].copy().reset_index()
    hsm_data['age_group'] = np.floor(hsm_data.age / 5) * 5
    hsm_data['age_group'] = hsm_data['age_group'].astype('int').astype('category')
    hsm_data = hsm_data.drop('age', axis = 1)

    hsm_data = hsm_data.groupby('age_group').sum()
    total = hsm_data.sum(axis =1)
    for col in hsm_data:
        hsm_data[col] = hsm_data[col] / total

    print total.sum()
    ax = hsm_data.plot(title = 'HSM - ' + sexe)
    fig = ax.get_figure()
    fig.savefig(os.path.join(assets_path, 'hsm_dependance_niveau_{}.png'.format(sexe)))

plot_data.query('sexe == 1').plot()



from scipy.special import erf as erf
value = -3 + 5
x = -2.96239829063 - value
.5 * (1.0 + erf(x / 1.41421356237))

hsm_data_bis

erf(-10000 / 1.41421356237)
