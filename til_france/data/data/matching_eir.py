'''
Created on 8 sept. 2014

@author: alexis
'''
import pandas as pd
import pdb

path_eic = 'D:\data\EIC\EIC 2009\\'

pdb.set_trace()

base = pd.read_csv(path_eic + 'EIC2009_base100.csv')

comp200 = pd.read_stata(path_eic + 'ano_eic200comp.dta')
# On recupere les individus cadre 
cadres = comp200['cc'] == '5000'
cadres_ind = comp200.loc[cadres, 'noind']
cadres_select = \
    comp200.loc[comp200['noind'].isin(cadres_ind),
                ['noind', 'cc', 'annee']]
    

test2 = comp.groupby(['noind'])['cc'].mean()