# -*- coding:utf-8 -*-
'''
Created on 11 septembre 2013

Ce programme :
Input :
Output :

'''

import logging
import os
from pandas import merge, DataFrame, concat, read_table
import numpy as np
import pdb
import time


# 1- Importation des classes/librairies/tables nécessaires à l'importation des données de Destinie -> Recup des infos dans Patrimoine

from data.DataTil import DataTil
from data.utils.utils import minimal_dtype, drop_consecutive_row
from pgm.CONFIG import path_data_destinie


log = logging.getLogger(__name__)


class Destinie(DataTil):

    def __init__(self):
        DataTil.__init__(self)
        self.name = 'Destinie'
        self.survey_year = 2009
        self.last_year = 2060
        self.survey_date = 100 * self.survey_year + 1
        # TODO: Faire une fonction qui check où on en est, si les précédent on bien été fait, etc.
        # TODO: Dans la même veine, on devrait définir la suppression des variables en fonction des étapes à venir.
        self.methods_order = [
            'load',
            'format_initial',
            'enf_to_par',
            'check_partner',
            'creation_menage',
            'creation_foy',
            'var_sup',
            'longitudinal',
            'add_futur',
            'store_to_liam'
            ]

    def _output_name(self):
        return 'Destinie.h5'

    def load(self):
        def _BioEmp_in_2():
            ''' Division de BioEmpen trois tables '''
            longueur_carriere = 106 # self.max_dur
            start_time = time.time()
            # TODO: revoir le colnames de BioEmp : le retirer ?
            colnames = list(range(longueur_carriere))
            path = os.path.join(path_data_destinie, 'BioEmp.txt')
            BioEmp = read_table(path, sep=';', header=None, names=colnames)
            taille = len(BioEmp) / 3
            BioEmp['id'] = BioEmp.index / 3

            # selection0 : informations atemporelles  sur les individus (identifiant, sexe, date de naissance et âge de fin d'étude)
            selection0 = [3 * x for x in range(taille)]
            ind = BioEmp.iloc[selection0].copy()
            ind.reset_index(inplace=True)
            ind.rename(columns = {1: 'sexe', 2: 'naiss', 3: 'findet', 4: 'tx_prime_fct'}, inplace=True)

            for column in ind.columns:
                if column in ['sexe', 'naiss', 'findet']:
                    ind[column] = ind[column].astype(int)
                elif column in ['tx_prime_fct']:
                    continue
                else:
                    del ind[column]

            ind['id'] = ind.index

            # selection1 : information sur les statuts d'emploi
            selection1 = [3 * x + 1 for x in range(taille)]
            statut = BioEmp.iloc[selection1].copy()
            statut = np.array(statut.set_index('id').stack().reset_index())
            #statut = statut.rename(columns={'level_1':'period', 0:'workstate'})
            #statut = statut[['id', 'period', 'workstate']] #.fillna(np.nan)
            #statut = minimal_dtype(statut)

            # selection2 : informations sur les salaires
            selection2 = [3 * x + 2 for x in range(taille)]
            sal = BioEmp.iloc[selection2].copy()
            sal = sal.set_index('id').stack().reset_index()
            sal = sal[0]
            #.fillna(np.nan)
            #sal = minimal_dtype(sal)

            # Merge de selection 1 et 2 :
            emp = np.zeros((len(sal), 4))
            emp[:, 0:3] = statut
            emp[:, 3] = sal
            emp = DataFrame(emp, columns=['id', 'period', 'workstate', 'salaire_imposable'])
            # Mise au format minimal
            emp = emp.fillna(np.nan).replace(-1, np.nan)
            emp = minimal_dtype(emp)
            return ind, emp

        def _lecture_BioFam():
            path = os.path.join(path_data_destinie, 'BioFam.txt')
            BioFam = read_table(path, sep=';',
                                   header=None, names=['id', 'pere', 'mere', 'civilstate', 'partner',
                                                       'enf1', 'enf2', 'enf3', 'enf4', 'enf5', 'enf6'])
            # Index limites pour changement de date
            delimiters = BioFam['id'].str.contains('Fin')
            annee = BioFam[delimiters].index.tolist()  # donne tous les index limites
            annee = [-1] + annee  # in order to simplify loops later
            # create a series period
            year0 = self.survey_year
            period = []
            for k in range(len(annee) - 1):
                period = period + [year0 + k] * (annee[k + 1] - 1 - annee[k])

            BioFam = BioFam[~delimiters].copy()
            BioFam['period'] = period
            list_enf = ['enf1', 'enf2', 'enf3', 'enf4', 'enf5', 'enf6']
            BioFam[list_enf + ['pere', 'mere', 'partner']] -= 1
            BioFam.loc[:, 'id'] = BioFam.loc[:, 'id'].astype(int) - 1
            for var in ['pere', 'mere', 'partner'] + list_enf:
                BioFam.loc[BioFam[var] < 0, var] = -1
            BioFam = BioFam.fillna(-1)
#             BioFam = drop_consecutive_row(BioFam.sort(['id', 'period']), ['id', 'pere', 'mere', 'partner', 'civilstate'])
            BioFam.replace(-1, np.nan, inplace=True)
            BioFam = minimal_dtype(BioFam)
            return BioFam

        print "Début de l'importation des données"
        start_time = time.time()
        self.ind, self.emp = _BioEmp_in_2()

        def _recode_sexe(sexe):
            ''' devrait etre dans format mais plus pratique ici'''
            if sexe.max() == 2:
                sexe.replace(1, 0, inplace=True)
                sexe.replace(2, 1, inplace=True)
            return sexe

        self.ind['sexe'] = _recode_sexe(self.ind['sexe'])
        self.BioFam = _lecture_BioFam()
        print "Temps d'importation des données : " + str(time.time() - start_time) + "s"
        print "fin de l'importation des données"

    def format_initial(self):
        '''
        Aggrégation des données en une seule base
            - ind : démographiques + caractéristiques indiv
            - emp_tot : déroulés de carrières et salaires associés
        '''
        print "Début de la mise en forme initiale"
        start_time = time.time()



        def _Emp_clean(ind, emp):
            ''' Mise en forme des données sur carrières:
            Actualisation de la variable période
            Création de la table décès qui donne l'année de décès des individus (index = identifiant)  '''
            emp = merge(emp, ind[['naiss']], left_on = 'id', right_on = ind[['naiss']].index)
            emp['period'] = emp['period'] + emp['naiss']
            #deces = emp.groupby('id')['period'].max()
            emp =  emp[['id', 'period', 'workstate', 'salaire_imposable']]

            # Séries de nouveaux états (non pris en compte pour l'instant)
            # contractuel + stagiaire -> RG non-cadre
            emp['workstate'].replace([11, 12, 13], 1)

            # maladie + invalidité  -> inactif
            emp['workstate'].replace([621, 623, 624, 63], 6)

            # Recodage des modalités
            # TO DO : A terme faire une fonction propre à cette étape -> _rename(var)
            # inactif   <-  1  # chomeur   <-  2   # non_cadre <-  3  # cadre     <-  4
            # fonct_a   <-  5  # fonct_s   <-  6   # indep     <-  7  # avpf      <-  8
            # preret    <-  9 #  décès, ou immigré pas encore arrivé en France <- 0
            # retraite <- 10 # etudiant <- 11 (scolarité hors cumul)
            emp['workstate'].replace(
                [0, 1, 2, 31, 32, 4, 5, 6, 7, 9, 8, 63],
                [0, 3, 4, 5, 6, 7, 2, 1, 9, 8, 10, 11],
                inplace = True
                )

            # Séries de nouveaux états (non pris en compte pour l'instant)
            return emp

        def _ind_total(BioFam, ind, emp):
            ''' fusion : BioFam + ind + emp -> ind '''
            survey_year = self.survey_year
            to_ind = merge(emp, BioFam, on=['id', 'period'], how ='left')
            ind = merge(to_ind, ind, on='id', how = 'left')
            ind.sort(['id', 'period'], inplace=True)
            cond_atemp = (
                (ind['naiss'] > survey_year) & (ind['period'] != ind['naiss'])
                ) | (
                (ind['naiss'] <= survey_year) & (ind['period'] != survey_year)
                )
            ind.loc[cond_atemp, ['sexe', 'naiss', 'findet', 'tx_prime_fct']] = -1
            return ind

        def _ind_in_3(ind):
            '''division de la table total entre les informations passées, à la date de l'enquête et futures
            ind -> past, ind, futur '''
            survey_year = self.survey_year
            ind_survey = ind.loc[ind['period'] == survey_year, :].copy()
            ind_survey.fillna(-1, inplace=True)
            if 'tx_prime_fct' in ind_survey.columns:
                ind_survey.rename(columns={'tx_prime_fct': 'tauxprime'}, inplace=True)
            print "Nombre dindividus présents dans la base en " + str(survey_year) + " : " + str(len(ind_survey))
            past = ind[ind['period'] < survey_year].copy()
            list_enf = ['enf1', 'enf2', 'enf3', 'enf4', 'enf5', 'enf6']
            list_intraseques = ['sexe', 'naiss', 'findet', 'tx_prime_fct']
            list_to_drop = list_intraseques + list_enf
            past.drop(list_to_drop, axis=1, inplace=True)
            self.longitudinal = past
            # past = drop_consecutive_row(past.sort(['id', 'period']), ['id', 'workstate', 'salaire_imposable'])
            log.info(
                u"Nombre de lignes sur le passé : " + str(len(past)) + " (informations de " +
                str(past['period'].min()) + u" à " + str(past['period'].max()) + ")"
                )

            # La table futur doit contenir une ligne par changement de statut à partir de l'année n+1, on garde l'année n, pour
            # voir si la situation change entre n et n+1
            # Indications de l'année du changement + variables inchangées -> -1
            futur = ind[ind['period'] >= survey_year].copy()
            futur.drop(list_enf, axis=1, inplace=True)
            futur.fillna(-1, inplace=True)
            # futur = drop_consecutive_row(futur.sort(['id', 'period']),
            #       ['id', 'workstate', 'salaire_imposable', 'pere', 'mere', 'civilstate', 'partner'])
            futur = futur[futur['period'] > survey_year]
            return ind_survey, past, futur

        def _work_on_futur(futur, ind):
            ''' ajoute l'info sur la date de décès '''
            # On rajoute une ligne par individu pour spécifier leur décès (seulement période != -1)

            def __deces_indicated_lastyearoflife():
                # dead = DataFrame(index = deces.index.values, columns = futur.columns)
                # dead['period'][deces.index.values] = deces.values
                # dead['id'][deces.index.values] = deces.index.values
                # dead.fillna(-1, inplace=True)
                # dead['death'] = dead['period']*100 + 1

                dead = DataFrame(deces)
                dead['id'] = dead.index
                dead['death'] = dead['period'] * 100 + 1

                futur = concat([futur, dead], axis=0, ignore_index=True)
                futur.fillna(-1, inplace=True)
                futur = futur.sort(['id', 'period', 'dead']).reset_index().drop('index', 1)
                futur.drop_duplicates(['id', 'period'], inplace=True)
                dead = futur[['id', 'period']].drop_duplicates('id', take_last=True).index
                futur['deces'] = -1
                futur.loc[dead, 'deces'] = 1
                futur = futur.sort(['period', 'id']).reset_index().drop(['index', 'dead'], 1)
                return futur

            def __death_unic_event(futur):
                futur = futur.sort(['id', 'period'])
                no_last = futur.duplicated('id', take_last=True)
                futur['death'] = -1
                cond_death = (not no_last) & ((futur['workstate'] == 0) | (futur['period'] != 2060))
                futur.loc[cond_death, 'death'] = 100 * futur.loc[cond_death, 'period'] + 1
                futur.loc[(futur['workstate'] != 0) & (futur['death'] != -1), 'death'] += 1
                add_lines = futur.loc[(futur['period'] > futur['death']) & (futur['death'] != -1), 'id']
                if len(add_lines) != 0:
                    # TODO: prévoir de rajouter une ligne quand il n'existe pas de ligne associée à la date de mort.
                    print len(add_lines)
                    pdb.set_trace()

                return futur

            futur = __death_unic_event(futur)

            # Types minimaux
            futur.replace(-1, np.nan, inplace=True)
            futur = minimal_dtype(futur)
            return futur

        emp = _Emp_clean(self.ind, self.emp)

        ind_total = _ind_total(self.BioFam, self.ind, emp)
        ind, past, futur = _ind_in_3(ind_total)
        futur = _work_on_futur(futur, ind)
        self.ind = ind
        self.past = past
        self.futur = futur
        print "Temps de la mise en forme initiale : " + str(time.time() - start_time) + "s"
        print "Fin de la mise en forme initiale"

    def enf_to_par(self):
        '''Vérifications des liens de parentés '''

        ind = self.ind
        list_enf = ['enf1', 'enf2', 'enf3', 'enf4', 'enf5', 'enf6']
        ind = ind.set_index('id')
        ind['id'] = ind.index
        year_ini = self.survey_year  # = 2009
        print "Début de l'initialisation des données pour " + str(year_ini)

        # Déclarations initiales des enfants
        pere_ini = ind[['id', 'pere']]
        mere_ini = ind[['id', 'mere']]
        list_enf = ['enf1', 'enf2', 'enf3', 'enf4', 'enf5', 'enf6']
        # Comparaison avec les déclarations initiales des parents
        for par in ['mere', 'pere']:
            # a -Définition des tables initiales:
            if par == 'pere':
                par_ini = pere_ini
                sexe = 0
            else:
                par_ini = mere_ini
                sexe = 1
            # b -> construction d'une table a trois entrées :
            #     par_decla = identifiant du parent déclarant l'enfant
            #     par_ini = identifiant du parent déclaré par l'enfant
            #     id = identifiant de l'enfant (déclaré ou déclarant)
            par_ini = par_ini[par_ini[par] != -1]
            link = ind.loc[(ind['enf1'] != -1) & (ind['sexe'] == sexe),  list_enf]
            link = link.stack().reset_index().rename(columns={'id': par, 'level_1': 'link', 0 :'id'})[[par,'id']].astype(int)
            link = link[link['id'] != -1]
            link = merge(link, par_ini, on = 'id', suffixes=('_decla', '_ini'),
                         how = 'outer').fillna(-1)
            link = link[(link[par + '_decla'] != -1) | (link[par + '_ini'] != -1)]
            ind['men_' + par] = 0

            # c- Comparaisons et détermination des liens
            # Cas 1 : enfants et parents déclarent le même lien : ils vivent ensembles
            parents = link.loc[(link[par + '_decla'] == link[par + '_ini']), 'id']
            ind.loc[parents.values, 'men_' + par] = 1

            # Cas 2 : enfants déclarant un parent mais ce parent ne les déclare pas (rattachés au ménage du parent)
            # Remarques : 8 cas pour les pères, 10 pour les mères
            parents = link[(link[par + '_decla'] != link[par + '_ini']) & (link[par + '_decla'] == -1)] ['id']
            ind.loc[parents.values, 'men_' + par] = 1
            print str(sum(ind['men_' + par] == 1)) + " vivent avec leur " + par

            # Cas 3 : parent déclarant un enfant mais non déclaré par l'enfant (car hors ménage)
            # Aucune utilisation pour l'instant (men_par = 0) mais pourra servir pour la dépendance
            parents = link.loc[(link[par + '_decla'] != link[par + '_ini']) & (link[par + '_ini'] == -1), ['id', par + '_decla']].astype(int)
            ind.loc[parents['id'].values, par] = parents[par + '_decla'].values
            print str(sum((ind[par].notnull() & (ind[par] != -1)))) + " enfants connaissent leur " + par

        self.ind = ind.drop(list_enf, axis=1)

    def creation_menage(self):
        ind = self.ind
        survey_year = self.survey_year
        ind['quimen'] = -1
        ind['men'] = -1
        ind['age'] = survey_year - ind['naiss']
        ind.fillna(-1, inplace=True)

        # 1ere étape : Détermination des têtes de ménages
        # (a) - Plus de 25 ans ou plus de 17ans ne déclarant ni pères, ni mères
        maj = (ind.loc[:, 'age'] >= 25) | ((ind.loc[:, 'men_pere'] == 0) & (
            ind.loc[:, 'men_mere'] == 0) & (ind.loc[:, 'age'] > 16))
        ind.loc[maj, 'quimen'] = 0
        print 'nb_sans_menage_a', len(ind.loc[~ind['quimen'].isin([0, 1]), :])

        # (b) - Personnes prenant en charge d'autres individus
        # Mères avec enfants à charge : (ne rajoute aucun ménage)
        enf_mere = ind.loc[(ind['men_pere'] == 0) & (ind['men_mere'] == 1) & (ind['age'] <= 25), 'mere'].astype(int)
        ind.loc[enf_mere.values, 'quimen'] = 0
        # Pères avec enfants à charge :(ne rajoute aucun ménage)
        enf_pere = ind.loc[(ind['men_mere'] == 0) & (ind['men_pere'] == 1) & (ind['age'] <= 25), 'pere'].astype(int)
        ind.loc[enf_pere.values, 'quimen'] = 0
        print 'nb_sans_menage_b', len(ind.loc[~ind['quimen'].isin([0, 1]), :])

        # (c) - Correction pour les personnes en couple non à charge [identifiant le plus petit = tête de ménage]
        ind.loc[(ind['partner'] > ind['id']) & (ind['partner'] != -1) & (ind['quimen'] !=- 2), 'quimen'] = 0
        ind.loc[(ind['partner'] < ind['id']) & (ind['partner'] != -1) & (ind['quimen'] !=- 2), 'quimen'] = 1
        print str(len(ind[ind['quimen'] == 0])) + u" ménages ont été constitués "  # 20815
        print "   dont " + str(len(ind[ind['quimen'] == 1])) + " couples "  # 9410

        # 2eme étape : attribution du numéro de ménage grâce aux têtes de ménage
        nb_men = len(ind.loc[(ind['quimen'] == 0), :])
        # Rq : les 10 premiers ménages correspondent à des institutions et non des ménages ordinaires
        # 0 -> DASS, 1 ->
        ind.loc[ind['quimen'] == 0, 'men'] = range(10, nb_men + 10)

        # 3eme étape : Rattachement des autres membres du ménage
        # (a) - Rattachements des partners des personnes en couples
        partner = ind.loc[(ind['quimen'] == 1), ['id', 'partner']].astype(int)
        ind['men'][partner['id'].values] = ind['men'][partner['partner'].values].copy()

        # (b) - Rattachements de leurs enfants (d'abord ménage de la mère, puis celui du père)
        for par in ['mere', 'pere']:
            enf_par = ind.loc[((ind['men_' + par] == 1) & (ind['men'] == -1)), ['id', par]].astype(int)
            ind['men'][enf_par['id']] = ind['men'][enf_par[par]]
            # print str(sum((ind['men']!= -1)))  + " personnes ayant un ménage attribué"

        # 4eme étape : création d'un ménage fictif résiduel :
        # Enfants sans parents :  dans un foyer fictif équivalent à la DASS = 0
        ind.loc[(ind['men'] == -1) & (ind['age'] < 18), 'men'] = 0

        # TODO: ( Quand on sera à l'étape gestion de la dépendance ) :
        # créer un ménage fictif maison de retraite + comportement d'affectation.
        ind['tuteur'] = -1
        # 5eme étape : mises en formes finales
        # attribution des quimen pour les personnes non référentes
        ind.loc[~ind['quimen'].isin([0, 1]), 'quimen'] = 2

        # suppressions des variables inutiles
        ind.drop(['men_pere', 'men_mere'], axis=1, inplace=True)

        # 6eme étape : création de la table men
        men = ind.loc[ind['quimen'] == 0, ['id', 'men']].copy()
        men.rename(columns={'id': 'pref', 'men': 'id'}, inplace=True)

        # Rajout des foyers fictifs
        to_add = DataFrame([np.zeros(len(men.columns))], columns = men.columns)
        to_add['pref'] = -1
        to_add['id'] = 0
        men = concat([men, to_add], axis = 0, join='outer', ignore_index=True)

        for var in ['loyer', 'tu', 'zeat', 'surface', 'resage', 'restype', 'reshlm', 'zcsgcrds', 'zfoncier', 'zimpot',
                    'zpenaliv', 'zpenalir', 'zpsocm', 'zrevfin']:
            men[var] = 0

        men['pond'] = 1
        men['period'] = survey_year
        men.fillna(-1, inplace=True)
        ind.fillna(-1, inplace=True)

        print ind[ind['men'] == -1].to_string()
        assert sum((ind['men'] == -1)) == 0  # Tout le monde a un ménage : on est content!
        assert sum((ind['quimen'] < 0)) == 0
        assert max(ind.loc[ind['quimen'] == 0, :].groupby('men')['quimen'].count()) == 1  # vérifie que le nombre de tête de ménage n'excède pas 1 par ménage
        print 'Taille de la table men :', len(men)
        self.ind = ind
        self.men = men

    def add_futur(self):
        print "Début de l'actualisation des changements jusqu'en 2060"
        # TODO: déplacer dans DataTil
        ind = self.ind
        futur = self.futur
        men = self.men
        foy = self.foy
        past = self.past

        # On précise les dates :
        for data in [ind, men, foy]:
            if data is not None:
                data['period'] = self.survey_year

        for data in [futur, past]:
            if data is not None:
                for var in ind.columns:
                    if var not in data.columns:
                        data[var] = -1

        # On ajoute ces données aux informations de 2009
        # TODO: être sur que c'est bien.
        ind = concat([ind, futur], axis=0, join='outer', ignore_index=True)
        ind.fillna(-1, inplace=True)
        men.fillna(-1, inplace=True)
        foy.fillna(-1, inplace=True)
        ind.sort(['period', 'id'], inplace=True)
        self.ind = ind
        self.men = men
        self.foy = foy
        print "Fin de l'actualisation des changements jusqu'en 2060"


if __name__ == '__main__':
    data = Destinie()
    start_t = time.time()
    # (a) - Importation des données et corrections préliminaires
    data.load()
    data.format_initial()

    # (b) - Travail sur la base initiale (données à l'année de l'enquête)
    ini_t = time.time()
    data.enf_to_par()
    data.check_partner()
    data.creation_menage()
    data.creation_foy()

    # (c) - Ajout des informations futures et mise au format Liam
    futur_t = time.time()
    # data.add_futur()
    data.format_to_liam()
    data.final_check()
    data.longitudinal_data()
    data.store_to_liam()
    log.info(
        "Temps Destiny.py : " + str(time.time() - start_t) + "s, dont " +
        str(futur_t - ini_t) + "s pour les mises en formes/corrections initiales et " +
        str(time.time() - futur_t) + "s pour l'ajout des informations futures et la mise au format Liam"
        )
