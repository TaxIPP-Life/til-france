# -*- coding:utf-8 -*-


from __future__ import division


import os
import pkg_resources


import numpy
import pandas


path_model = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )

uniform_weight = 200
year = 2010
error_margin = .01


# Data from INSEE projections
data_path = '/home/benjello/openfisca/Til-BaseModel/til_base_model/param/demo/projpop0760_FECcentESPcentMIGcent.xls'
sheetname_by_gender = dict(zip(
    ['total', 'male', 'female'],
    ['populationTot', 'populationH', 'populationF']
    ))
population_insee_by_gender = dict(
    (
        gender,
        pandas.read_excel(data_path, sheetname = sheetname, skiprows = 2, header = 2)[:109].set_index(
            u'Âge au 1er janvier')
        )
    for gender, sheetname in sheetname_by_gender.iteritems()
    )
populationF = population_insee_by_gender['female'][year]
populationH = population_insee_by_gender['male'][year]
population = population_insee_by_gender['total'][year]

# Data from INSEE patrimoine survey
patrimoine_insee_path = "/home/benjello/data/til/patrimoine"
individus_patrimoine = pandas.read_csv(
    os.path.join(patrimoine_insee_path, 'individu.csv')
    )[['sexe', 'age', 'pond', 'identmen']]
menages_patrimoine = pandas.read_csv(
    os.path.join(patrimoine_insee_path, 'menage.csv')
    )
outre_mer = menages_patrimoine.loc[menages_patrimoine['zeat'] == 0, 'identmen'].copy()
individus_patrimoine = individus_patrimoine[~individus_patrimoine['identmen'].isin(outre_mer)].copy()

individus_patrimoine['age_group'] = individus_patrimoine.age // 10

# Data form TIL (patrimoine after reworking)
patrimoine_til_path = os.path.join(path_model, 'Patrimoine_next_metro_{}.h5'.format(uniform_weight))
store = pandas.HDFStore(patrimoine_til_path)
individus_til = store['/entities/individus']
individus_til['age'] = individus_til.age_en_mois // 12
individus_til['age_group'] = individus_til.age // 10

# Data form TIL (destine before/after reworking)
destinie_til_path = os.path.join(path_model, 'Destinie.h5')
store = pandas.HDFStore(destinie_til_path)
store.close()
store.open()
destinie_til = store['/entities/individus']
destinie_til['age'] = individus_til.age_en_mois // 12
destinie_til['age_group'] = individus_til.age // 10

insee_proj_pop_F = populationF.sum()
insee_proj_pop_H = populationH.sum()
insee_proj_pop_total = insee_proj_pop_F + insee_proj_pop_H

destinie_uniform_weight = insee_proj_pop_total / (destinie_til.age >= 0).sum()


print 'population totale INSEE: ', insee_proj_pop_total
print 'population totale Patrimoine (hors DOM): ', ((individus_patrimoine.age >= 0) * individus_patrimoine.pond).sum()
print 'population totale TIL patrimoine: ', (individus_til.age >= 0).sum() * uniform_weight
print 'population totale TIL Destinie: ', (destinie_til.age >= 0).sum() * destinie_uniform_weight


print 'Ecart absolu: ', (individus_til.age >= 0).sum() * uniform_weight - insee_proj_pop_total
print 'Ecart absolu hommes: ', ((individus_til.age >= 0) & (individus_til.sexe == 0)).sum() * uniform_weight - insee_proj_pop_H
print 'Ecart absolu femmes: ', ((individus_til.age >= 0) & (individus_til.sexe == 1)).sum() * uniform_weight - insee_proj_pop_F


print 'Ecart absolu: ', (destinie_til.age >= 0).sum() * destinie_uniform_weight - insee_proj_pop_total
print 'Ecart absolu hommes: ', ((destinie_til.age >= 0) & (destinie_til.sexe == 0)).sum() * destinie_uniform_weight- insee_proj_pop_H
print 'Ecart absolu femmes: ', ((destinie_til.age >= 0) & (destinie_til.sexe == 1)).sum() * destinie_uniform_weight - insee_proj_pop_F


# Grouping by age_group

# INSEE projections
population.index.name = 'age'
population = population.reset_index(name = 'total')
population.age = population.index
population['age_group'] = population.age // 10
population = population[['age_group', 'total']].groupby('age_group').sum()

# Patrimoin survey data
population_patrimoine = individus_patrimoine.loc[
    individus_patrimoine.age >= 0, ['age_group', 'pond']
    ].groupby('age_group').sum().apply(numpy.round)
population_patrimoine.rename(columns = dict(pond = 'total_pat'), inplace = True)


# TIL data
population_til = individus_til.loc[individus_til.age >= 0, ['age_group', 'age']].groupby('age_group').count() * uniform_weight
population_til.rename(columns = dict(age = 'total_til'), inplace = True)

# TIL data
population_destinie = destinie_til.loc[destinie_til.age >= 0, ['age_group', 'age']].groupby('age_group').count() * destinie_uniform_weight
population_destinie.rename(columns = dict(age = 'total_destinie'), inplace = True)


import matplotlib as plt
population_til
population
summary =  pandas.concat([population_patrimoine, population_til, population_destinie, population], axis = 1)
summary = summary.append(summary.sum(), ignore_index = True)

summary.eval('abs_diff_til = total_til - total', engine = 'python')
summary.eval('rel_diff_til = abs_diff_til / total', engine = 'python')

summary.eval('abs_diff_destinie = total_destinie - total', engine = 'python')
summary.eval('rel_diff_destinie = abs_diff_destinie / total', engine = 'python')

summary.plot(kind = 'bar')
