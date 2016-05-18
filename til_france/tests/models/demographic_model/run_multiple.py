# -*- coding: utf-8 -*-

from __future__ import division


import os
import pandas
import shutil


from liam2.simulation import Simulation
# from liam2.main_multiple_run import simulate_replications


def simulate_replications(fpath, output_path, replications):
    print("Using simulation file: '{}' to run {} replications".format(fpath, replications))
    for seed in range(replications):
        print(seed)
        output_dir = os.path.join(output_path, 'replication' + '_' + str(seed))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        print(output_path)
        simulation = Simulation.from_yaml(fpath, output_dir = output_dir, seed = seed)

        simulation.run(False)


fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'console_unaligned.yml')
output_path = replications_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

replications = 30
simulate_replications(fpath, output_path, replications)

replication_directories = [
    os.path.join(replications_directory, d) for d in os.listdir(replications_directory)
    if os.path.basename(d).startswith('replication_')]


dataframe_by_replication = dict()
for replication_directory in replication_directories:
    dataframe = pandas.read_csv(os.path.join(replication_directory, 'deces.csv'), header = None)
    dataframe.rename(columns = {0: 'annee', 1: 'deces'}, inplace = True)
    dataframe['replication'] = os.path.basename(replication_directory)
    dataframe_by_replication[os.path.basename(replication_directory)] = dataframe

df = pandas.concat(dataframe_by_replication.values())

pivoted = pandas.pivot_table(df, values='deces', columns='replication', index='annee')

ax = pivoted.plot(style = 'r-', legend = None)
pivoted.mean(axis=1).plot(style = 'b--', ax = ax)

if os.path.exists(os.path.join(output_path, 'deces.csv')):
    os.remove(os.path.join(output_path, 'deces.csv'))


fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'console.yml')
Simulation.from_yaml(fpath).run(False)

aligned = pandas.read_csv(os.path.join(output_path, 'deces.csv'), header = None)
aligned = aligned.rename(columns = {0: 'annee', 1: 'deces'}).set_index('annee')
aligned.plot(style = 'g-', ax = ax)
