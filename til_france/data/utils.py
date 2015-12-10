# -*- coding:utf-8 -*-
from __future__ import print_function

'''
Created on 2 août 2013
@author: a.eidelman
'''

import logging
import numpy as np
from pandas import Series, DataFrame
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import pdb


log = logging.getLogger(__name__)


of_name_to_til = {
    'individus': 'individus',
    'foyers_fiscaux': 'foyers_fiscaux',
    'menages': 'menages',
    'familles': 'familles',
    'futur': 'futur',
    'past': 'past'
    }


def new_idmen(table, var):
    new = table[[var + '_ini', var]].copy()
    men_ord = (table[var + '_ini'] > 9)
    men_nonord = (table[var + '_ini'] < 10)
    # les ménages nonordinaires gardent leur identifiant initial même si leur pondération augmente
    new.loc[men_nonord, var] = new.loc[men_nonord, var + '_ini']
    # on conserve la règle du début des identifiants à 10 pour les ménages ordinaires
    new.loc[men_ord, var] = range(10, 10 + len(men_ord))
    new = new[var]
    return new


def new_link_with_men(table, table_exp, link_name):
    '''
    A partir des valeurs initiables de lien initial (link_name) replique pour avoir le bon nouveau lien
    '''
    nb_by_table = np.asarray(table.groupby(link_name).size())
    # TODO: améliorer avec numpy et groupby ?
    group_old_id = table_exp.loc[
        table_exp['id_ini'].isin(table[link_name]),
        ['id_ini', 'id'],
        ].groupby('id_ini').groups.values()
    group_old_id = np.array(group_old_id)
    group_old_id = group_old_id.repeat(nb_by_table)
    new_id = []
    for el in group_old_id:
        # log.info(el)
        new_id += el
    return new_id

