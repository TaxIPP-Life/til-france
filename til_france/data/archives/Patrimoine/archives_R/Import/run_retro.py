# -*- coding:utf-8 -*-

'''
Created on 2 Apr 2013

@author: alexis_e
'''

chemin = 'M:\\Myliam2\\'
chemin = 'C:\\Myliam2\\'

import sys
sys.path.append(chemin+'src')

import main 
from simulation import Simulation

########### RETRO ############
fichier = chemin+'Patrimoine\\Past\\import.yml'

sys.argv.append('import')
sys.argv.append(fichier)
main.main()
sys.argv.remove(fichier)

fichier = chemin+'Patrimoine\\lien_parent_enfant\\match_score.yml'
fichier = chemin+'Patrimoine\\Past\\retro.yml'

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

