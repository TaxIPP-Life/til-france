# -*- coding:utf-8 -*-
'''
Created on 25 Apr 2013

@author: alexis_e
'''

from pandas import HDFStore, merge, DataFrame
import numpy as np
import pdb
import time
import os

from til import __path__ as path_til
from utils import of_name_to_til, concatenated_ranges

def table_for_of(simulation, period=None, check_validity=False, save_tables=False):
    temps = time.clock()
    output_tab = os.path.join(path_til[0], "output", "to_run_leg.h5" )
    # on travaille d'abord sur l'ensemble des tables puis on selectionne chaque annee
    # on étudie d'abord la table individu pour pouvoir séléctionner les identifiants
    # step 1
    table = {}

    entities = simulation.entities
    entities_name =  map( lambda e: e.name, simulation.entities)
    def _get_entity(name):
        position = entities_name.index(name)
        return simulation.entities[position]

    ind = _get_entity('person')
    table['ind'] = DataFrame(ind.array.columns)
    table['ind'] = table['ind'].rename(columns={'men': 'idmen', 'idfoy': 'idfoy', 'id': 'noi', 'statmarit': 'civilstate'})

    # création de variable
    table['ind']['ageq'] = table['ind']['age']/5 - 4
    table['ind']['ageq'] = table['ind']['ageq']*(table['ind']['ageq'] > 0)
    table['ind']['ageq'] = 12 + (table['ind']['ageq']-12)*(table['ind']['ageq'] < 12)
    #TODO: modifier pour les jeunes veufs

    # create fam entity
    try:
        table['ind'][['idfam','quifam']] = table['ind'].loc[:,['idmen','quimen']]
    except:
        pdb.set_trace()

#    # Travail sur les qui quand on ne controle pas dans la simulation que tout le monde n'est pas qui==2
## inutile car fait maintenant dans la simulation mais peut-être mieux à refaire ici un jour
## parce que ça prend du temps dans la simulation
#    time_qui = time.clock()
#    for ent in ('men','foy'): # 'fam' un jour...
#        print "Deal with qui for ", ent
#        qui= 'qui'+ent
#        ident = 'id'+ent
#        trav = table['ind'].ix[table['ind'][qui]==2, [ident,qui,'period']]
#        for name, groupfor nom in ('menage','declar','fam'):for nom in ('menage','declar','fam'): in trav.groupby([ident,'period']):
#            to_add = range(len(group))
#            group[qui] = group[qui]+to_add
#            table['ind'].ix[group[qui].index, qui] = group[qui]
#        print "les qui pour ", ent," sont réglés"
#    time_qui = time.clock() - time_qui
#    print "le temps passé à s'occuper des qui a été",time_qui
    ind = table['ind']
    for ent in ['men', 'foy']:
        entity = _get_entity(of_name_to_til[ent])

        table[ent] = DataFrame(entity.array.columns)
        id = 'id' + ent
        qui = 'qui' + ent
        table[ent] = table[ent].rename(columns={'id': id})
        # travail sur les qui
        nb_qui = ind.loc[ind[qui]>1, ['noi',id,qui]].groupby(id, sort=True).size()
        if len(nb_qui)>0:
            new_qui = concatenated_ranges(nb_qui) + 2
            table['ind'] = table['ind'].sort(id) #note the sort
            col_qui = table['ind'][qui]
            col_qui[col_qui>1] = new_qui
            table['ind'][qui] = col_qui


        # informations on qui == 0
        qui0 = table['ind'].loc[table['ind']['qui' + ent]==0,['noi','idfoy','idmen','idfam','period']]
        table[ent] = merge(table[ent], qui0, how='left', left_on=[id,'period'], right_on=[id,'period'])

        if ent=='men':
            # nbinde est limité à 6 personnes et donc valeur = 5 en python
            table[ent]['nbinde'] = (table[ent]['nb_persons']-1) * (table[ent]['nb_persons']-1 <=5) +5*(table[ent]['nb_persons']-1 >5)
            table['fam'] = qui0

    # remove non-ordinary household
    cond = (table['ind']['idmen'] >= 10) & (table['ind']['idfoy'] >= 10)
    table['ind'] = table['ind'][cond]
    table['men'] = table['men'][table['men']['idmen']>=10]
    table['idfoy'] = table['idfoy'][table['idfoy']['idfoy']>=10]
    table['fam'] = table['fam'][table['fam']['idfam']>=10]
    # get years
    years = np.unique(table['ind']['period'].values/100)
    if period is not None:
        years=[period]
        print years

    if check_validity:
        for year in years:
            ind = table['ind']
            for ent in ['men', 'foy']: #fam
                id = 'id' + ent
                qui = 'qui' + ent
                tab = table[ent]
                try:
                    assert ind.groupby([id,qui]).size().max() == 1
                except:
                    print ent
                    pb = ind.groupby([id,qui]).size() > 1
                    print(ind.groupby([id,qui]).size()[pb])
                    pdb.set_trace()
                    print(ind[ind[id]==43][['noi',id,qui]])

                qui0 = ind[ind[qui]==0]
                try:
                    assert qui0[id].isin(tab[id]).all()
                except:
                    cond = tab[id].isin(qui0[id])
                    print(tab[~cond])
                    pdb.set_trace()
                try:
                    assert tab[id].isin(qui0[id]).all()
                except:
                    cond = tab[id].isin(qui0[id])
                    print(tab[~cond])
                    pdb.set_trace()

    for year in years:
        if save_tables:
            try:
                os.remove(output_tab)
            except:
                print("Attention, la table intermediaire n'a pas ete supprimee")
            goal = HDFStore(output_tab)
            goal.remove('survey_'+str(year))
            for ent in ('ind', 'men', 'foy','fam'):
                tab = table[ent].loc[table[ent]['period']/100==year]
                key = 'survey_'+str(year) + '/'+ent
                goal.put(key, tab)
            goal.close()
        else:
            for ent in ('ind', 'men', 'foy', 'fam'):
                table[ent] = table[ent].loc[table[ent]['period']/100==year]
            return table

if __name__ == "__main__":
    table_for_of()
