# -*- coding:utf-8 -*-
'''
Created on 25 juil. 2013

@author: a.eidelman
'''

import pandas as pd
import pdb
import numpy as np
import time

# #TODO: J'aime bien l'idée de regarder sur les valeurs potentielles, de séléctionner la meilleure et de prendre quelqu'un dans cette case.
# # L'avantage c'est qu'au lieu de regarder sur tous les individus, on regarde sur les valeurs, on a potentiellement plusieurs gens par 
# # case donc moins de case que de gens ce qui peut améliorer le temps de calcul. 
# age_groups = table2['age'].unique()
# dip_groups = table2['diploma'].unique()
# group_combinations = [(ag, dip) for ag in age_groups for dip in dip_groups]
# 
# def iter_key(age_grp, dip_grp, age, dip):
#     age_av = sum(age_grp) / 2.0
#     dip_av = sum(dip_grp) / 2.0
#     return pow(age - age_av, 2) + pow(dip - dip_av, 2)
# 
# def iterate_buckets(age, dip):
#     combs = sorted(group_combinations)
#     for c in combs:
#         yield c
#         
# def match_key(indiv1, indiv2):
#     age1, dip1 = indiv1
#     age2, dip2 = indiv2
#     return pow(age1 - age2, 2) + pow(dip1 - dip2, 2)
# 
# pdb.set_trace()
# 
# 
# # def get_best_match(matches, age, dip):
# #     sorted_matches = sorted(key=match_key, zip(matches, [(age, dip)] * len(match)))
# #     return sorted_matches[0]
# 
# pdb.set_trace()
# 
# for indiv in table1: 
#     print indiv
# 
# for individual in table1:
#     age, diploma = individual
#     for age_bucket, dip_bucket in iterate_buckets(table2['age'], table2['diploma']):
#         matches = age_bucket.intersection(dip_bucket)
#         if matches:
#             match = get_best_match(matches, age, diploma)
#             all_matches.append((individual, match))
#             remove_individual(age_groups, match)
#             remove_individual(dip_groups, match)


    
# matching()
# import cProfile
# command = """matching()"""
# cProfile.runctx( command, globals(), locals(), filename="matching.profile1" )

#### temps de calcul en fonction de la base
def run_time(n):
    table2 = pd.DataFrame(np.random.randint(0,100, [n,2]), columns=['age','diploma'])
    table1 = pd.DataFrame(np.random.randint(0,100, [n,2]), columns=['age','diploma'])
 
    score_str = "(table2['age']-temp['age'])**2 +  5*(table2['diploma']-temp['diploma'])"
    
    match = pd.Series(0, index=table1.index)
    index2 = pd.Series(True, index=table2.index)  
    k_max = min(len(table2), len(table1))
    
    debut = time.clock()
    for k in xrange(k_max):   
        temp = table1.iloc[k] 
        score = eval(score_str)
        score = score[index2]
        idx2 = score.idxmax()
        match.iloc[k] = idx2 # print( k, 0, index2)
        index2[idx2] = False
    print 'taille: ',n,' ; temps de calcul: ', time.clock()-debut
    return time.clock()-debut


def run_time_cell(n):
    table2 = pd.DataFrame(np.random.randint(0,100, [n,2]), columns=['age','diploma'])
    table1 = pd.DataFrame(np.random.randint(0,100, [n,2]), columns=['age','diploma'])
 
    match = pd.Series(0, index=table1.index)
    index2 = pd.Series(True, index=table2.index)  
    k_max = min(len(table2), len(table1))
 
    age_groups = table2['age'].unique()
    dip_groups = table2['diploma'].unique()
    group_combinations = np.array([[ag, dip] for ag in age_groups for dip in dip_groups])
    groups2 = table2.groupby(['age','diploma'])

    cell_values = pd.DataFrame(groups2.groups.keys())
    temp = pd.DataFrame(groups2.size())
    temp = temp.rename(columns={0:'nb'})
    cell_values = cell_values.merge(temp, left_on=[0,1], right_index=True)
    
    score_str = "(cell_values[0]-temp['age'])**2 +  5*(cell_values[1]-temp['diploma'])"
       
    debut = time.clock()
    for k in xrange(len(table1)):   
        temp = table1.iloc[k] 
        score = eval(score_str)
        idx2 = score.idxmax()
        match.iloc[k] = idx2 # print( k, 0, index2)
        cell_values.loc[idx2,'nb'] -= 1
        if cell_values.loc[idx2,'nb']==0:
            cell_values = cell_values.drop(idx2, axis=0)

    print 'taille: ',n,' ; temps de calcul: ', time.clock()-debut
    return time.clock()-debut


def run_time_np(n):
    table2 = np.random.randint(0,100, [n,2])
    table1 = np.random.randint(0,100, [n,2])
    idx2 = np.array([np.arange(n)])
    table2 = np.concatenate((table2, idx2.T), axis=1)
    
    match = np.empty(n, dtype=int)
    k_max = min(len(table2), len(table1))
    score_str = "(table2[:,0]-temp[0])**2 +  5*(table2[:,1]-temp[1])"
    k_max = min(len(table2), len(table1))
    debut = time.clock()
    for k in xrange(k_max):   
        temp = table1[k]
        score = eval(score_str)
        idx = score.argmax()
        idx2 = table2[idx,2]
        match[k] = idx2 
        table2 = np.delete(table2, idx, 0)
    print 'taille: ',n,' ; temps de calcul: ', time.clock()-debut
    return time.clock()-debut
        
def run_time_np_cell(n):
  
    table2 = pd.DataFrame(np.random.randint(0,100, [n,2]), columns=['age','diploma'])
    table1 = pd.DataFrame(np.random.randint(0,100, [n,2]), columns=['age','diploma'])
 
    match = pd.Series(0, index=table1.index)
    index2 = pd.Series(True, index=table2.index)  
    k_max = min(len(table2), len(table1))
 
    age_groups = table2['age'].unique()
    dip_groups = table2['diploma'].unique()
    group_combinations = np.array([[ag, dip] for ag in age_groups for dip in dip_groups])
    groups2 = table2.groupby(['age','diploma'])

    cell_values = pd.DataFrame(groups2.groups.keys())
    temp = pd.DataFrame(groups2.size())
    temp = temp.rename(columns={0:'nb'})
    cell_values = cell_values.merge(temp, left_on=[0,1], right_index=True)
    cell_values['idx'] = range(len(cell_values))
    
    
    table1 = np.array(table1)
    cell_values = np.array(cell_values)
    
    match = np.empty(n, dtype=int)
    score_str = "(cell_values[:,0]-temp[0])**2 +  5*(cell_values[:,1]-temp[1])"
    k_max = len(table1)
    debut = time.clock()
    for k in xrange(k_max):   
        temp = table1[k]
        score = eval(score_str)
        idx = score.argmax()
        idx2 = cell_values[idx,3]
        match[k] = idx2 
        cell_values[idx,2] -= 1
        if cell_values[idx,2]==0:
            cell_values = np.delete(cell_values, idx, 0)
    print 'taille: ',n,' ; temps de calcul: ', time.clock()-debut
    pdb.set_trace()
    return time.clock()-debut        

temps = {}
sizes = [500000,1500000,2000000,1000000,2500000]
#[1500000,2000000,1000000,2500000]
#[20000,25000,30000,35000,40000,45000,50000,75000,100000]
# [1000,3000,5000,7000,8000,10000,12500,15000]
        #20000,25000,30000,35000,40000,45000,50000,75000,100000

for size in sizes:
    temps[str(size)] = run_time_np_cell(size)
#     
# for size in sizes:
#     temps[str(size)] = run_time(size)    
#     
    
    
