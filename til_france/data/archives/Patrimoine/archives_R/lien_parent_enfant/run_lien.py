# -*- coding:utf-8 -*-

'''
Created on 2 Apr 2013

@author: alexis_e
'''
import sys 
sys.path.append('C:\\til-liam\\')
sys.path.append('C:\\openfisca\\')

from src_liam.simulation import Simulation
from src_liam.importer import  file2h5
#from src_liam.simulation import Simulation

########### LIEN ############
chemin = 'C:\\til\\'
fichier = chemin+'data\\Patrimoine\\lien_parent_enfant\\import.yml'

file2h5(fichier)


fichier = chemin+'data\\Patrimoine\\lien_parent_enfant\\match_score.yml'

simulation= Simulation.from_yaml(
                                 fichier,
                     input_dir=None,
                    input_file=None,
                    output_dir=None,                    
                    output_file=None)
simulation.run(False)


fichier = chemin+'Patrimoine\\lien_parent_enfant\\match_par_enf.yml'
fichier = chemin+'Patrimoine\\lien_parent_enfant\\expand.yml'
fichier = chemin+'Patrimoine\\lien_parent_enfant\\match_par_enf.yml'
fichier = chemin+'Patrimoine\\lien_parent_enfant\\match_score.yml'

