# -*- coding:utf-8 -*-


import os
import sys


import logging
import numpy as np
from pandas import concat, DataFrame, merge, read_csv, read_hdf, Series

from openfisca_core.calmar import calmar


from til_core.config import Config
from til_core.data.utils.matching import Matching
from til_core.data.utils.misc import recode, minimal_dtype
from til_france.data.DataTil import DataTil
from til_france.targets.population import get_data_frame_insee


log = logging.getLogger(__name__)


class Patrimoine(DataTil):

    def __init__(self):
        DataTil.__init__(self)
        self.name = 'Patrimoine'
        self.survey_year = 2009
        self.last_year = 2009
        # self.survey_date = 100 * self.survey_year + 1
        self.survey_date = self.survey_year
        # TODO: Faire une fonction qui check où on en est, si les précédent on
        # bien été fait, etc.
        # TODO: Dans la même veine, on devrait définir la suppression des
        # variables en fonction des étapes à venir.
        self.methods_order = [
            'load',
            'drop_variable',
            'to_DataTil_format',
            'champ',
            'correction',
            'partner',
            'enfants',
            'expand_data',
            'creation_child_out_of_house',
            'matching_par_enf',
            'matching_couple_hdom',
            'creation_foy',
            'mise_au_format',
            'var_sup',
            'store_to_liam'
            ]

# drop_variable() doit tourner avant table_initial() car on aurait un problème
# avec les variables qui sont renommées.
# explication de l'ordre en partant de la fin, besoin des couples pour et des
# liens parents enfants pour les mariages.
# Ces liens ne peuvent se faire qu'après la dupplication pour pouvoir avoir
# le bon nombre de parents et de bons matchs
# La dupplication, c'est mieux si elle se fait après la création de
# child_out_of_house, plutôt que de chercher à créer child_out_of_house à
# partir de la base étendue
# Pour les enfants, on cherche leur parent un peu en fonction de s'ils sont en
# couple ou non, ça doit donc touner après partner.
# Ensuite, c'est assez évident que le format initial et le drop_variable
# doivent se faire le plus tôt possible
# on choisit de faire le drop avant le format intitial, on pourrait faire
# l'inverse en étant vigilant sur les noms

    def load(self):
        log.info(u"Début de l'importation des données")
        config = Config()
        patrimoine_data_directory = config.get('raw_data', 'patrimoine_data_directory')
        path_ind = os.path.join(patrimoine_data_directory, 'individu.csv')
        individus = read_csv(path_ind)
        path_men = os.path.join(patrimoine_data_directory, 'menage.csv')
        menages = read_csv(path_men)
        individus_institutions_path = config.get('raw_data', 'hsi_data_directory')
        individus_institutions = read_hdf(os.path.join(individus_institutions_path, 'hsi_extract.h5'))

        individus['identmen'] = individus['identmen'].apply(int)
        menages['identmen'] = menages['identmen'].apply(int)
        log.info(
            u"Nombre de ménages dans l'enquête initiale : " +
            str(len(menages['identmen'].drop_duplicates()))
            )
        log.info(
            u"Nombre d'individus dans l'enquête initiale : " +
            str(len(individus['identind'].drop_duplicates()))
            )

        self.entity_by_name['menages'] = menages
        self.entity_by_name['individus'] = individus
        self.entity_by_name['individus_institutions'] = individus_institutions

        assert (menages['identmen'].isin(individus['identmen'])).all()
        assert (individus['identmen'].isin(menages['identmen'])).all()
        log.info(u"fin de l'importation des données")

    def champ(self, option='metropole'):
        u''' Limite la base à un champ d'étude défini '''
        assert option in ['metropole']
        individus = self.entity_by_name['individus']
        menages = self.entity_by_name['menages']
        # TODO: enable multplie restriction (option is a list)
        if option == 'metropole':
            #  Se place sur le champ France métropolitaine en supprimant les
            # antilles parce qu'elles n'ont pas les même variables et que
            # l'appariemment EIR n'est pas possible
            antilles = menages.loc[menages['zeat'] == 0, 'id']
            menages = menages[~menages['id'].isin(antilles)]
            individus = individus[~individus['idmen'].isin(antilles)]

        self.entity_by_name['individus'] = individus
        self.entity_by_name['menages'] = menages

    def to_DataTil_format(self):
        individus = self.entity_by_name['individus']
        menages = self.entity_by_name['menages']
        individus_institutions = self.entity_by_name['individus_institutions']

        til_name_by_patrimoine_name = {
            'zsalaires_i': 'salaire_imposable',
            'zchomage_i': 'choi',
            'zpenalir_i': 'alr',
            'zretraites_i': 'rsti',
            'anfinetu': 'findet',
            'etamatri': 'civilstate',
            'cyder': 'anc',
            'duree': 'xpr',
            }
        individus.rename(columns=til_name_by_patrimoine_name, inplace=True)
        # id, men
        menages.index = range(10, len(menages) + 10)
        menages['id'] = menages.index
        individus['id'] = individus.index
        idmen = menages[['id', 'identmen']].rename(columns={'id': 'idmen'})
        individus = merge(individus, idmen, on='identmen')

        individus['period'] = self.survey_date
        menages['period'] = self.survey_date
        # age_en_mois
        age = self.survey_year - individus['anais']
        individus['age_en_mois'] = 12 * age + 11 - individus['mnais']

        individus['sexe'].replace([1, 2], [0, 1], inplace=True)
        individus['civilstate'].replace([2, 1, 4, 3, 5], [1, 2, 3, 4, 5], inplace=True)
        individus.loc[individus['pacs'] == 1, 'civilstate'] = 5
        # workstate
        # Code DataTil : {inactif: 1, chomeur: 2, non_cadre: 3, cadre: 4,
        # fonct_a: 5, fonct_s: 6, indep: 7, avpf: 8, preret: 9}
        individus['workstate'] = individus['statut'].replace(
            [1, 2, 3, 4, 5, 6, 7], [6, 6, 3, 3, 1, 7, 7]
            )
        # AVPF
        # TODO: l'avpf est de la législation, ne devrait pas être un statut de workstate
        cond_avpf = (menages['paje'] == 1) | (menages['complfam'] == 1) | \
                    (menages['allocpar'] == 1) | (menages['asf'] == 1)
        avpf = menages.loc[cond_avpf, 'id']
        individus.loc[(individus['idmen'].isin(avpf)) & (individus['workstate'].isin([1, 2])), 'workstate'] = 8
        # cadre, non cadre
        individus.loc[(individus['classif'].isin([6, 7])) & (individus['workstate'] == 5), 'workstate'] = 6
        individus.loc[(individus['classif'].isin([6, 7])) & (individus['workstate'] == 3), 'workstate'] = 4
        # Pas très bon car actif, sedentaire et pas cadre non cadre

        # retraite
        individus.loc[individus['preret'] == 1, 'workstate'] = 9
        individus.loc[(individus['anais'] < self.survey_year - 64) & (individus['workstate'] == 1), 'workstate'] = 10
        individus['workstate'].fillna(1, inplace=True)
        individus['workstate'] = individus['workstate'].astype(np.int8)

        # findet
        individus['findet'].replace(0, np.nan, inplace=True)
        individus['findet'] = individus['findet'] - individus['anais']
        # tauxprime
        individus['tauxprime'] = 0
        # dependance_niveau
        individus['dependance_niveau'] = -1

        # Individus en institutions
        individus_institutions['id'] = range(
            individus.id.max() + 1, individus.id.max() + 1 + len(individus_institutions))
        individus_institutions['idmen'] = range(
            individus.idmen.max() + 1, individus.idmen.max() + 1 + len(individus_institutions))

        individus['data_origin'] = 0
        individus_institutions['data_origin'] = 1
        age = self.survey_year - individus_institutions['anais']
        individus_institutions['age_en_mois'] = 12 * age + 11 - individus_institutions['mnais'].values
        individus_institutions['lienpref'] = 0
        individus_institutions['workstate'] = 1  # inactif
        individus_institutions['period'] = self.survey_date
        individus_institutions['civilstate'] = 2  # on les forces à célibataires TODO leur attirbuer un partner
        individus_institutions['dependance_niveau'] = (
            2 * individus_institutions['rgir'].isin([1, 2]) + 1 * individus_institutions['rgir'].isin([3, 4])
            ) + 0
        log.info(u'Ajout de {} individus en institution aux {} individus initiaux'.format(
            len(individus_institutions), len(individus)))
        individus = individus.append(individus_institutions, verify_integrity = True, ignore_index = True)
        log.info(u'La table individus contient désormais {} individus'.format(len(individus)))

        individus_institutions_menages = individus_institutions[['idmen', 'pond']].copy()
        individus_institutions_menages.rename(columns = {'idmen': 'id'}, inplace = True)
        individus_institutions_menages['period'] = self.survey_date
        log.info(u'Ajout de {} menages en institution aux {} ménages initiaux'.format(
            len(individus_institutions_menages), len(menages)))
        menages = menages.append(individus_institutions_menages, verify_integrity = True, ignore_index = True)
        log.info(u'La table menages contient désormais {} ménages'.format(len(menages)))

        self.entity_by_name['individus'] = individus
        self.entity_by_name['menages'] = menages
        self.entity_by_name['individus_institutions'] = individus_institutions

        self.drop_variable({
            'menages': ['identmen', 'paje', 'complfam', 'allocpar', 'asf'],
            'individus': ['identmen', 'preret']
            })

    def corrections(self):
        pass

    def work_on_past(self, method='from_data'):
        assert method in ['from_external_match', 'from_data']
        individus = self.entity_by_name['individus']

        def _correction_carriere(metro = True):
            '''
            Fait des corrections sur le déroulé des carrières
            ( à partir de vérif écrit en R)
            '''
            # Note faire attention à la numérotation à partir de 0
            # TODO: faire une verif avec des asserts
            individus['cydeb1'] = individus['prodep']
            liste1 = [6723, 7137, 10641, 21847, 30072, 31545, 33382]
            liste1 = list(set(liste1).intersection(set(individus.index)))
            liste1 = [x - 1 for x in liste1]
            individus['cydeb1'][liste1] = individus.anais[liste1] + 20
            individus['cydeb1'][15206] = 1963
            individus['cydeb1'][27800] = 1999
            individus['modif'] = ""
            individus['modif'].iloc[liste1 + [15206, 27800]] = "cydeb1_manq"

            individus['cyact3'][10833] = 4
            individus['cyact2'][23584] = 11
            individus['cyact3'][27816] = 5
            individus['modif'].iloc[[10833, 23584, 27816]] = "cyact manq"
            var = ["cyact", "cydeb", "cycaus", "cytpto"]
            # Note : la solution ne semble pas être parfaite au sens qu'elle ne résout pas tout
            # cond : gens pour qui on a un probleme de date
            cond0 = (individus['cyact2'].notnull()) & (individus['cyact1'].isnull()) & (
                (individus.cydeb1 == individus.cydeb2) |
                (individus.cydeb1 > individus.cydeb2) |
                (individus.cydeb1 == (individus.cydeb2 - 1))
                )
            cond0.iloc[8297] = True
            individus['modif'][cond0] = "decal act"
            # on decale tout de 1 à gauche en espérant que ça résout le problème
            for k in range(1, 16):
                var_k = [x + str(k) for x in var]
                var_k1 = [x + str(k + 1) for x in var]
                individus.loc[cond0, var_k] = individus.loc[cond0, var_k1]

            # si le probleme n'est pas resolu, le souci était sur cycact seulement, on met une valeur
            cond1 = individus['cyact2'].notnull() & individus['cyact1'].isnull() & (
                (individus['cydeb1'] == individus['cydeb2']) |
                (individus['cydeb1'] > individus['cydeb2']) |
                (individus['cydeb1'] == (individus['cydeb2'] - 1))
                )
            individus.loc[cond1, 'modif'] = "cyact1 manq"
            individus.loc[cond1 & (individus['cyact2'] != 4), 'cyact1'] = 4
            individus.loc[cond1 & (individus['cyact2'] == 4), 'cyact1'] = 2

            cond2 = individus['cydeb1'].isnull() & (individus['cyact1'].notnull() | individus['cyact2'].notnull())
            individus.loc[cond2, 'modif'] = "jeact ou anfinetu manq"
            individus['findet_year'] = individus['findet'] + individus['anais']
            individus.loc[cond2, 'cydeb1'] = individus.loc[cond2, ['jeactif', 'findet_year']].max(axis = 1)
            # quand l'ordre des dates n'est pas le bon on fait l'hypothèse que
            # c'est la première date entre anfinetu et jeactif qu'il faut prendre en non pas l'autre
            cond2 = individus['cydeb1'] > individus['cydeb2']
            individus.loc[cond2, 'cydeb1'] = individus.loc[cond2, ['jeactif', 'findet_year']].min(axis=1)
            return individus

        # travail sur les carrières
        if method == 'from_external_match':
            path_patr_past = os.path.join(patrimoine_data_directory, 'carriere_passee_patrimoine.csv')
            past = read_csv(path_patr_past)
            assert past['identind'].isin(individus['identind']).all()
            # TODO: it's hard-coded
            past_years = range(1980, 2010)
            dates = [100 * year + 1 for year in past_years]
            salaire_imposable = DataFrame(columns=dates)
            workstate = DataFrame(columns=dates)
            for year in past_years:
                workstate[100 * year + 1] = past['statut' + str(year)]
                salaire_imposable[100 * year + 1] = past[
                    ['indep_tot' + str(year), 'cadre_tot' + str(year), 'chom_tot_brut' + str(year)]
                    ].sum(axis=1)

            # TODO: add id in longitudinal
            workstate['identind'] = past['identind']
            workstate = individus[['identind']].merge(workstate, on=['identind'], how='left')
            workstate.fillna(-1, inplace=True)
            workstate.drop('identind', axis=1, inplace=True)
            self.longitudinal['workstate'] = workstate
            salaire_imposable['identind'] = past['identind']
            salaire_imposable = individus[['identind']].merge(salaire_imposable, on=['identind'], how='left')
            salaire_imposable.fillna(-1, inplace=True)
            salaire_imposable.drop('identind', axis=1, inplace=True)
            self.longitudinal['salaire_imposable'] = salaire_imposable

        if method == 'from_data':
            ind = _correction_carriere()
            # travail sur les carrières
            survey_year = self.survey_year
            date_deb = int(min(individus['cydeb1']))
            n_ind = len(ind)
            calend = np.zeros((n_ind, survey_year - date_deb), dtype=int)

            nb_even = range(16)
            cols_deb = ['cydeb' + str(i + 1) for i in nb_even]
            tab_deb = individus[cols_deb].fillna(0).astype(int).values
            cols_act = ['cyact' + str(i + 1) for i in nb_even]
            tab_act = np.empty((n_ind, len(nb_even) + 1), dtype=int)
            tab_act[:, 0] = -1
            tab_act[:, 1:] = individus[cols_act].fillna(0).astype(int).values

            idx = range(n_ind)
            col_idx = np.zeros(n_ind, dtype=int)
            # c'est la colonne correspondant à l'indice de la prochaine date
            # comme tab_act est décalé de 1, c'est aussi l'indice de la situation en cours
            for year in range(date_deb, survey_year):
                to_change = (tab_deb[idx, col_idx] == year) & (col_idx < 15)
                col_idx[to_change] += 1
                calend[:, year - date_deb] = tab_act[idx, col_idx]
            colnames = [100 * year + 1 for year in range(date_deb, survey_year)]

            self.longitudinal['workstate'] = DataFrame(calend, columns=colnames)
            # TODO: imputation for salaire_imposable
            self.longitudinal['salaire_imposable'] = 0 * self.longitudinal['workstate']

        all = self.entity_by_name['individus'].columns.tolist()
        carriere = [x for x in all if x[:2] == 'cy' and x not in ['cyder', 'cysubj']] + ['jeactif', 'prodep']
        self.drop_variable(dict_to_drop={'individus': carriere + ['identind', 'noi']})

    def drop_variable(self, dict_to_drop=None, option='white'):
        '''
        - Si on dict_to_drop is not None, il doit avoir la forme table: [liste de variables],
        on retire alors les variable de la liste de la table nommée.
        - Sinon, on se sert de cette méthode pour faire la première épuration des données, on
         a deux options:
             - passer par la liste blanche ce que l'on recommande pour l'instant
             - passer par  liste noire.
        '''
        individus = self.entity_by_name['individus']
        menages = self.entity_by_name['menages']

        if dict_to_drop is None:
            dict_to_drop = {}

        # travail sur ménages
            all = menages.columns.tolist()
            # liste noire
            pr_or_cj = [
                x for x in all if (x[-2:] == 'pr' or x[-2:] == 'cj') and x not in ['indepr', 'r_dcpr', 'r_detpr']]
            detention = [x for x in all if len(x) == 6 and x[0] == 'p' and x[1] in ['0', '1']]
            diplom = [x for x in all if x[:6] == 'diplom']
            partner_died = [x for x in all if x[:2] == 'cj']
            even_rev = [x for x in all if x[:3] == 'eve']
            black_list = pr_or_cj + detention + diplom + partner_died + even_rev  # + enfants_hdom
            # liste blanche
            var_to_declar = ['zcsgcrds', 'zfoncier', 'zimpot', 'zpenaliv',
                             'zpenalir', 'zpsocm', 'zrevfin']
            var_apjf = ['asf', 'allocpar', 'complfam', 'paje']
            enfants_hdom = [x for x in all if x[:3] == 'hod']
            white_list = ['id', 'identmen', 'pond', 'period'] + var_apjf + enfants_hdom + var_to_declar
            if option == 'white':
                dict_to_drop['menages'] = [x for x in all if x not in white_list]
            else:
                dict_to_drop['menages'] = black_list

        # travail sur individus
            all = individus.columns.tolist()
            # liste noire
            parent_prop = [x for x in all if x[:6] == 'jepro_']
            jeunesse_grave = [x for x in all if x[:6] == 'jepro_']
            jeunesse = [x for x in all if x[:7] == 'jegrave']
            black_list = jeunesse_grave + parent_prop + diplom
            # liste blanche
            info_pers = ['anais', 'mnais', 'sexe', 'dip14', 'age_en_mois', 'findet', 'tauxprime']
            famille = ['couple', 'lienpref', 'enf', 'civilstate', 'pacs', 'grandpar', 'per1e', 'mer1e', 'enfant']
            jobmarket = ['statut', 'situa', 'workstate', 'preret', 'classif', 'cs42']
            info_parent = ['jepnais', 'jemnais', 'jemprof']
            dependance = ['data_origin', 'dependance_niveau']
            carriere = [x for x in all if x[:2] == 'cy' and x not in ['cyder', 'cysubj']] + \
                ['jeactif', 'anfinetu', 'prodep']
            revenus = ["zsalaires_i", "zchomage_i", "zpenalir_i", "zretraites_i", "cyder", "duree"]
            white_list = ['identmen', 'idmen', 'noi', 'pond', 'id', 'identind', 'period'] + info_pers + famille + \
                jobmarket + carriere + info_parent + revenus + dependance

            if option == 'white':
                dict_to_drop['individus'] = [x for x in all if x not in white_list]
            else:
                dict_to_drop['individus'] = black_list
        DataTil.drop_variable(self, dict_to_drop, option)

    def partner(self):
        ''' Calcul de l'identifiant du conjoint/partner et corrige les statuts '''
        # ne gère pas les identifiants des couples non cohabitants, on n'a pas l'info.
        log.info(u"Travail sur les partners")
        individus = self.entity_by_name['individus']
        # pour simplifier on créer une variable lien qui à la même valeur pour les personnes
        # potentielleùent en couple
        # Elle est plus pratique que lienpref (lien avec la personne de référence)
        # Pour les individus en institutions on a initialisé lienpref à 0
        individus['lien'] = individus['lienpref'].replace([1, 31, 32, 50], [0, 2, 3, 10])
        # Les gens mariés ou pacsés sont considéré en couple par définition (pas d'union factice):
        # reminder couple:
        # 1 Oui, avec une personne qui vit dans le logement
        # 2 Oui, avec une personne qui ne vit pas dans le logement
        # 3 Non
        problematic_couple = (individus['civilstate'].isin([1, 5])) & (individus['couple'] == 3)
        # individus.groupby(['civilstate', 'couple']).size()
        if sum(problematic_couple):
            problematic_individuals = individus.loc[
                problematic_couple, ['idmen', 'couple', 'civilstate', 'lienpref', 'lien', 'id']
                ]
            # si deux se disent non en couple mais mariés, on corrige avec couple = 1
            # sauf s'ils sont tous deux enfants de la même pref on met couple = 2 (cas de idmen = 6622)
            prob_by_menages = problematic_individuals['idmen'].value_counts()
            many_by_menages = prob_by_menages.loc[prob_by_menages > 1].index.values
            for idmen in many_by_menages:
                indices = problematic_individuals.loc[problematic_individuals.idmen == idmen].index
                if (problematic_individuals.loc[indices, 'lienpref'] == 2).all():
                    individus.loc[indices, 'couple'] = 2
                else:
                    individus.loc[indices, 'couple'] = 1

            # si une personne est seule mariés ou pacsé dans le ménage, on met aussi couple=1
            potential_partner_cond = individus['civilstate'].isin([1, 5]) & individus['idmen'].isin(
                problematic_individuals['idmen']) & (individus['couple'] == 1)
            potential_partner = individus.loc[potential_partner_cond, ['idmen', 'lien']]
            to_correct = problematic_individuals[['idmen', 'lien', 'id']].merge(potential_partner, how='inner')
            individus.loc[to_correct['id'].values, 'couple'] = 1
            # sinon, on met couple=2
            # on regarde si on a un lienpref =1, 31 ou 32 qui sont bien synonymes de couple
            update = (individus['civilstate'].isin([1, 5])) & (individus['couple'] == 3)
            individus.loc[update, 'couple'] = 2

        # Change les personnes qui se disent en couple en couple hors domicile
        # (on pourrait mettre non en couple aussi)
        cond_hdom = individus[individus['couple'] == 1].groupby(['idmen', 'lien']).size() == 1

        to_change = cond_hdom[cond_hdom].reset_index()
        to_change = merge(individus[individus['couple'] == 1], to_change, how='inner')

        individus.loc[to_change['id'].values, 'couple'] = 2

        # On ne cherche que les identifiants des couple vivant ensemble.
        assert sum(~individus[individus['couple'] == 1].groupby(['idmen', 'lien']).size() == 2) == 0
        in_couple = individus.loc[individus['couple'] == 1, ['idmen', 'lien', 'id', 'civilstate']]
        couple = in_couple.merge(in_couple, on=['idmen', 'lien'], suffixes=('', '_partner'))
        couple = couple[couple['id'] != couple['id_partner']]
        assert len(couple) == len(in_couple)
        individus['partner'] = -1
        individus.loc[couple['id'].values, 'partner'] = couple['id_partner'].values
        # On accorde les civilstate
        # Note: priority to marriages. -> 50 cases on 16 000
        # TODO: test that hypothesis on final restults
        # couple.loc[couple['civilstate_partner'] == 2, 'civilstate'].value_counts()
        couple.loc[couple['civilstate_partner'] == 1, 'civilstate'] = 1
        couple.loc[(couple['civilstate_partner'] == 5) & (couple['civilstate'] != 1), 'civilstate'] = 5
        individus.loc[couple['id'].values, 'civilstate'] = couple['civilstate'].values

        # Absence de polygames (seul -1 est partner de plusieurs individus)
        assert list(individus.partner.value_counts()[individus.partner.value_counts() > 1].index) == [-1]

        log.info(u"Fin du travail sur les partners")
        self.entity_by_name['individus'] = individus

    def enfants(self):
        '''
        Calcule l'identifiant des parents
        '''
        # la variable enf précise le lien de parenté entre l'enfant et les personnes de référence du ménage:
        # 1: enfant de la pref et de son conjoint/partner,
        # 2: enfant de pref seulement,
        # 3: enf de partner seulement,
        # 4: partner de l'enf de pref/partner

        individus = self.entity_by_name['individus']
        info_par = individus.loc[:, ['idmen', 'lienpref', 'id', 'sexe']]
        var_to_keep = ['id', 'idmen', 'sexe']

        # [0] Enfants de la personne de référence
        enf_pref = individus.loc[individus['enf'].isin([1, 2]), var_to_keep]
        enf_pref = enf_pref.merge(
            info_par[info_par['lienpref'] == 0], on = ['idmen'], how = 'left', suffixes = ('_enf', '_par'))

        # [1] Enfants du conjoint/partner de la personne de référence
        enf_partner = individus.loc[individus['enf'].isin([1, 3]), var_to_keep]
        enf_partner = enf_partner.merge(
            info_par[info_par['lienpref'] == 1], on = ['idmen'], suffixes = ('_enf', '_par'))

        # Parents de la personne de référence
        par_pref = individus.loc[individus['lienpref'] == 3, var_to_keep]
        par_pref = par_pref.merge(info_par[info_par['lienpref'] == 0],
                                  on=['idmen'], suffixes=('_par', '_enf'))

        # Beaux-Parents de la personne de référence
        bo_par_pref = individus.loc[individus['lienpref'] == 32, var_to_keep]
        bo_par_pref = bo_par_pref.merge(info_par[info_par['lienpref'] == 1],
                                        on=['idmen'], suffixes=('_par', '_enf'))

        # Grand -parents de la personne de référence (2 cas)
        gpar_pref = individus.loc[individus['lienpref'] == 22, var_to_keep]
        gpar_pref = gpar_pref.merge(info_par[info_par['lienpref'] == 3],
                                    on=['idmen'], suffixes=('_par', '_enf'))

        # Petits-enfants de la personne de référence
        petit_enfant = individus.loc[individus['lienpref'] == 21, var_to_keep]
        par_petit_enfant = individus.loc[
            (individus['enf'].isin([1, 2, 3])) & (individus['enfant'] == 2),
            ['idmen', 'lienpref', 'id', 'sexe', 'age_en_mois', 'enfant']
            ]
        par_petit_enfant.drop_duplicates('idmen', inplace=True)
        # TODO: en toute rigueur, il faudrait garder un lien si on ne trouve pas les parents pour l'envoyer dans le
        # registre...
        # et savoir que ce sont les petites enfants (pour l'héritage par exemple), pareil pour grands-parents quand
        # parents inconnus
        petit_enfant = merge(petit_enfant, par_petit_enfant, on=['idmen'], suffixes=('_enf', '_par'))

        linked = enf_pref.append([enf_partner, par_pref, bo_par_pref, gpar_pref, petit_enfant])
        assert linked.groupby(['id_enf', 'sexe_par']).size().max() == 1
        linked_pere = linked.loc[linked['sexe_par'] == 0]
        individus.loc[linked_pere['id_enf'].values, 'pere'] = linked_pere['id_par'].values
        linked_mere = linked.loc[linked['sexe_par'] == 1]
        individus.loc[linked_mere['id_enf'].values, 'mere'] = linked_mere['id_par'].values

        individus['pere'].fillna(-1, inplace=True)
        individus['mere'].fillna(-1, inplace=True)
        assert -1 in individus['pere'].values
        assert -1 in individus['mere'].values
        self._check_links(individus)

        # frere soeur
        sibblings = individus.loc[individus['lienpref'] == 10, ['idmen', 'id']]
        sibblings = sibblings.merge(
            individus.loc[individus['lienpref'] == 0, ['pere', 'mere', 'idmen']], on='idmen', how='inner')
        individus.loc[sibblings['id'].values, 'pere'] = sibblings['pere'].values
        individus.loc[sibblings['id'].values, 'mere'] = sibblings['mere'].values
        self._check_links(individus)
        # Last call, find the parent when we know he or she is there
        look_mother = individus.loc[
            (individus['mer1e'] == 1) & (individus['mere'] == -1),
            ['idmen', 'lienpref', 'id', 'age_en_mois']
            ]
        look_mother = look_mother[look_mother['lienpref'].isin([1, 2])]
        potential_mother = individus.loc[
            (individus['sexe'] == 1) & (~individus['lienpref'].isin([0, 2, 3])),
            ['id', 'idmen', 'sexe', 'age_en_mois', 'lienpref']
            ]
        potential_mother = potential_mother[~potential_mother['id'].isin(look_mother['id'])]
        match_mother = look_mother.merge(potential_mother, on=['idmen'], suffixes=('_enf', '_par'))
        match_mother.sort_values(
            by = ['idmen', 'lienpref_par', 'age_en_mois_par'],
            ascending = [True, False, False],
            inplace = True
            )
        match_mother.drop_duplicates('id_enf', keep='first', inplace=True)
        individus.loc[match_mother['id_enf'].values, 'mere'] = match_mother['id_par'].values
        self._check_links(individus)
        # TODO: Find a better affectation rule
        look_father = individus.loc[
            (individus['per1e'] == 1) & (individus['pere'] == -1),
            ['idmen', 'lienpref', 'id', 'age_en_mois']
            ]
        look_father = look_father[look_father['lienpref'].isin([1, 2])]
        potential_father = individus.loc[
            (individus['sexe'] == 0) & (~individus['lienpref'].isin([0, 2, 3])),
            ['id', 'idmen', 'sexe', 'age_en_mois', 'lienpref']
            ]
        potential_father = potential_father[~potential_father['id'].isin(look_father['id'])]
        match_father = look_father.merge(potential_father, on=['idmen'], suffixes=('_enf', '_par'))
        match_father.sort_values(
            by = ['idmen', 'lienpref_par', 'age_en_mois_par'],
            ascending = [True, False, False],
            inplace = True
            )
        match_father.drop_duplicates('id_enf', keep='first', inplace=True)
        individus.loc[match_father['id_enf'].values, 'pere'] = match_father['id_par'].values
        self._check_links(individus)
        log.info('Nombre de mineurs sans parents : {}'.format(
            sum((individus['pere'] == -1) & (individus['mere'] == -1) & (individus['age_en_mois'] < 12 * 18))
            ))
        test = (individus['pere'] == -1) & (individus['mere'] == -1) & (individus['age_en_mois'] < 12 * 18)
        individus.loc[test, ['lienpref', 'mer1e', 'per1e']]
        parent_trop_jeune = individus.loc[(individus['age_en_mois'] < 12 * 17), 'id']

        self._check_links(individus)
        assert sum((individus['pere'].isin(parent_trop_jeune)) | (individus['mere'].isin(parent_trop_jeune))) == 0
        self.entity_by_name['individus'] = individus

    def creation_child_out_of_house(self):
        u'''
        Renvoie une table qui doit se lire comme étant les enfants hors foyer déclarés par le ménage.
        On marque les infos que l'on connait sur ces enfants.
        On ajouter les infos sur leurs parents (qui sont donc des membres du ménage)

        On fera ensuite un matching de ces enfants avec les enfants qui ne vivent pas avec leur parent alors
        que ceux-ci sont vivants.
        '''
        individus = self.entity_by_name['individus']
        menages = self.entity_by_name['menages']
        # création brute de enfants hors du domicile
        child_out_of_house = DataFrame()
        for k in range(1, 13):
            k = str(k)
            # hodln : lien de parenté
            var_hod = ['hodln', 'hodsex', 'hodan', 'hodco', 'hodip', 'hodenf', 'hodemp', 'hodcho', 'hodpri', 'hodniv']
            var_hod_rename = ['hodln', 'sexe', 'anais', 'couple', 'dip6', 'nb_enf', 'hodemp', 'hodcho', 'hodpri',
                'hodniv']
            var_hod_k = [var + k for var in var_hod]
            temp = menages.loc[menages[var_hod_k[0]].notnull(), ['id'] + var_hod_k]
            dict_rename = {'id': 'idmen'}
            for num_varname in range(len(var_hod_rename)):
                dict_rename[var_hod_k[num_varname]] = var_hod_rename[num_varname]
            temp.rename(columns=dict_rename, inplace=True)

            temp['situa'] = Series(dtype=np.int8)
            temp.loc[temp['hodemp'] == 1, 'situa'] = 1
            temp.loc[temp['hodemp'] == 2, 'situa'] = 5
            temp.loc[temp['hodcho'] == 1, 'situa'] = 4
            temp.loc[temp['hodcho'] == 2, 'situa'] = 6
            temp.loc[temp['hodcho'] == 3, 'situa'] = 3
            temp.loc[temp['hodcho'] == 4, 'situa'] = 7

            temp['classif'] = Series()
            prive = temp['hodpri'].isin([1, 2, 3, 4])
            temp.loc[prive, 'classif'] = temp.loc[prive, 'hodpri']
            temp.loc[~prive, 'classif'] = temp.loc[~prive, 'hodniv']

            child_out_of_house = child_out_of_house.append(temp)
            len_ini = len(child_out_of_house)

        var_parent = ["id", "idmen", "sexe", "anais", "cs42", "grandpar"]
        # Si les parents disent qu'ils ont eux-même des parents vivants,
        # c'est que les grands parents de leurs enfants sont vivants !
        individus['grandpar'] = individus['per1e'].isin([1, 2]) | individus['mer1e'].isin([1, 2])

        # info sur les personnes de référence et leur partner
        info_pr = individus.loc[(individus['lienpref'] == 0), var_parent]
        info_cj = individus.loc[(individus['lienpref'] == 1), var_parent]

        # répartition entre père et mère en fonction du sexe...
        info_pr_pere = info_pr[info_pr['sexe'] == 0].rename(columns = {
            'id': 'pere',
            'anais': 'jepnais',
            'grandpar': 'grandpar_pat',
            'cs42': 'jepprof',
            'sexe': 'to_delete'
            })
        info_cj_pere = info_cj[info_cj['sexe'] == 0].rename(columns = {
            'id': 'pere',
            'anais': 'jepnais',
            'grandpar': 'grandpar_pat',
            'cs42': 'jepprof',
            'sexe': 'to_delete'
            })
        # ... puis les meres
        info_pr_mere = info_pr[info_pr['sexe'] == 1].rename(columns = {
            'id': 'mere',
            'anais': 'jemnais',
            'grandpar': 'grandpar_mat',
            'cs42': 'jemprof',
            'sexe': 'to_delete'
            })
        info_cj_mere = info_cj[info_cj['sexe'] == 1].rename(columns = {
            'id': 'mere',
            'anais': 'jemnais',
            'grandpar': 'grandpar_mat',
            'cs42': 'jemprof',
            'sexe': 'to_delete'
            })
        info_pere = info_pr_pere.append(info_cj_pere)
        info_mere = info_pr_mere.append(info_cj_mere)

        # A qui est l'enfant ?
        # aux deux

        cond1 = child_out_of_house['hodln'] == 1
        child_out_of_house1 = merge(child_out_of_house[cond1], info_pere, on='idmen', how='left')
        child_out_of_house1 = merge(child_out_of_house1, info_mere, on='idmen', how = 'left')

        # à la pref
        cond2 = child_out_of_house['hodln'] == 2
        child_out_of_house2 = merge(child_out_of_house[cond2], info_pr_pere, on='idmen', how='left')
        child_out_of_house2 = merge(child_out_of_house2, info_pr_mere, on='idmen', how = 'left')
        # au partner
        cond3 = child_out_of_house['hodln'] == 3
        child_out_of_house3 = merge(child_out_of_house[cond3], info_cj_pere, on='idmen', how='left')
        child_out_of_house3 = merge(child_out_of_house3, info_cj_mere, on='idmen', how = 'left')

        # len(temp) = len(child_out_of_house) - 4 #deux personnes du même sexe qu'on a écrasé a priori.
        child_out_of_house = concat(
            [child_out_of_house1, child_out_of_house2, child_out_of_house3], axis = 0, ignore_index = True)

        assert child_out_of_house.pere.isnull().any(), u"Les pères manquants ne sont pas repérés par des NaN"
        assert child_out_of_house.mere.isnull().any(), u"Les mères manquantes ne sont pas repérés par des NaN"

        # TODO: il y a des ménages avec hodln = 1 et qui pourtant n'ont pas deux membres
        # (à moins qu'ils aient le même sexe).
        # child_out_of_house = child_out_of_house.drop(
        #    ['hodcho', 'hodemp', 'hodniv', 'hodpri', 'to_delete_x', 'to_delete_y', 'jepprof'],axis=1)

        assert child_out_of_house['jemnais'].max() < 2010 - 18
        assert child_out_of_house['jepnais'].max() < 2010 - 18

        for parent in ['pere', 'mere']:
            check = child_out_of_house.merge(
                individus[['id', 'anais', 'age_en_mois', 'sexe', 'idmen', 'partner', 'pere', 'mere', 'lienpref']],
                left_on=parent, right_on='id', how='left', suffixes=('', '_' + parent)
                )
            diff_age = check['anais'] - check['anais_' + parent]
            child_out_of_house = child_out_of_house[((diff_age > 15) | (diff_age.isnull())).values]

        assert child_out_of_house.pere.isnull().any(), u"Les pères manquants ne sont pas repérés par des NaN"
        assert child_out_of_house.mere.isnull().any(), u"Les mères manquantes ne sont pas repérés par des NaN"

        self.child_out_of_house = child_out_of_house.fillna(-1)

    def matching_par_enf(self):
        u'''
        Matching des parents et des enfants hors du domicile
        '''
        log.info(u"Début du matching des parents et des enfants hors du domicile")
        individus = self.entity_by_name['individus']
        individus = individus.fillna(-1)
        individus.index = individus['id']
        child_out_of_house = self.child_out_of_house
        # info sur les parents hors du domicile des enfants
        cond_enf_look_par = (individus['per1e'] == 2) | (individus['mer1e'] == 2)
        enf_look_par = individus[cond_enf_look_par].copy()
        # Remarque: avant on mettait à zéro les valeurs quand on ne cherche pas le parent, maintenant
        # on part du principe qu'on fait les choses assez minutieusement
        enf_look_par['dip6'] = recode(
            enf_look_par['dip14'],
            [[30, 5], [41, 4], [43, 3], [50, 2], [60, 1]],
            method='geq'
            )
        enf_look_par['classif'] = recode(
            enf_look_par['classif'],
            [[[1, 2, 3], 4], [[4, 5], 2], [[6, 7], 1], [[8, 9], 3], [[10], 0]],
            method='isin'
            )
        # nb d'enfant
        # -- Au sein du domicile
        nb_enf_mere_dom = individus.groupby('mere').size()
        nb_enf_pere_dom = individus.groupby('pere').size()
        # On assemble le nombre d'enfants pour les peres et meres en enlevant les manquantes ( = -1)
        enf_tot_dom = concat([nb_enf_mere_dom, nb_enf_pere_dom], axis=0)
        enf_tot_dom = enf_tot_dom.drop([-1])   # -1 is in the index (pere or mere = -1)
        # -- Hors domicile
        nb_enf_mere_hdom = child_out_of_house.groupby('mere').size()
        nb_enf_pere_hdom = child_out_of_house.groupby('pere').size()

        enf_tot_hdom = concat([nb_enf_mere_hdom, nb_enf_pere_hdom], axis = 0)
        enf_tot_hdom = enf_tot_hdom.drop([-1])

        enf_tot = concat([enf_tot_dom, enf_tot_hdom], axis = 1).fillna(0)
        enf_tot = enf_tot[0] + enf_tot[1]
        # Sélection des parents ayant des enfants (enf_tot) à qui on veut associer des parents (enf_look_par)
        enf_tot = (enf_tot.loc[enf_tot.index.isin(enf_look_par.index)].astype(int)).copy()
        enf_look_par.index = enf_look_par['id']
        enf_look_par['nb_enf'] = 0
        enf_look_par.loc[enf_tot.index.values, 'nb_enf'] = enf_tot
        # Note: Attention le score ne peut pas avoir n'importe quelle forme, il faut des espaces devant les mots,
        # à la limite une parenthèse
        var_match = ['jepnais', 'situa', 'nb_enf', 'anais', 'classif', 'couple', 'dip6', 'jemnais', 'jemprof', 'sexe']
        # TODO: gerer les valeurs nulles, pour l'instant c'est très moche

        # TODO: avoir une bonne distance, on met un gros coeff sur l'age sinon, on a des parents,
        # plus vieux que leurs enfants
        score = "- 1000 * (other.anais - anais) **2 - 1.0 * (other.situa - situa) **2 " + \
            "- 0.5 * (other.sexe - sexe) **2 - 1.0 * (other.dip6 - dip6) **2 " + \
            " - 1.0 * (other.nb_enf - nb_enf) **2"

        # etape1 : deux parents vivants
        cond1_enf = (enf_look_par['per1e'] == 2) & (enf_look_par['mer1e'] == 2)
        cond1_par = (child_out_of_house['pere'] != -1) & (child_out_of_house['mere'] != -1)
        # TODO: si on fait les modif de variables plus tôt, on peut mettre directement child_out_of_house1
        # à cause du append plus haut, on prend en fait ici les premiers de child_out_of_house
        match1 = Matching(
            enf_look_par.loc[cond1_enf, var_match],
            child_out_of_house.loc[cond1_par, var_match],
            score
            )
        parent_found1 = match1.evaluate(
            orderby=['anais'],
            method='cells'
            )
        individus.loc[parent_found1.index.values, ['pere', 'mere']] = \
            child_out_of_house.loc[parent_found1.values, ['pere', 'mere']]

        # etape 2 : seulement mère vivante
        enf_look_par.loc[parent_found1.index, ['pere', 'mere']] = \
            child_out_of_house.loc[parent_found1, ['pere', 'mere']]
        cond2_enf = (enf_look_par['mere'] == -1) & (enf_look_par['mer1e'] == 2)

        cond2_par = np.logical_and(
            np.logical_not(child_out_of_house.index.isin(parent_found1)),
            child_out_of_house['mere'] != -1
            )
        match2 = Matching(
            enf_look_par.loc[cond2_enf, var_match],
            child_out_of_house.loc[cond2_par, var_match],
            score
            )
        parent_found2 = match2.evaluate(orderby = None, method='cells')
        individus.loc[parent_found2.index, ['mere']] = child_out_of_house.loc[parent_found2, ['mere']]

        # étape 3 : seulement père vivant
        enf_look_par.loc[parent_found2.index, ['pere', 'mere']] = child_out_of_house.loc[
            parent_found2, ['pere', 'mere']]
        cond3_enf = ((enf_look_par['pere'] == -1)) & (enf_look_par['per1e'] == 2)
        cond3_par = ~child_out_of_house.index.isin(parent_found1) & (child_out_of_house['pere'] != -1)

        # TODO: changer le score pour avoir un lien entre pere et mere plus évident
        match3 = Matching(enf_look_par.loc[cond3_enf, var_match],
                          child_out_of_house.loc[cond3_par, var_match], score)
        parent_found3 = match3.evaluate(orderby=None, method='cells')
        individus.loc[parent_found3.index, ['pere']] = child_out_of_house.loc[parent_found3, ['pere']]

        log.info(u" au départ on fait " + str(len(parent_found1) + len(parent_found2) + len(parent_found3)) +
            " match enfant-parent hors dom")
        # on retire les match non valides
        to_check = individus[['id', 'age_en_mois', 'sexe', 'idmen', 'partner', 'pere', 'mere', 'lienpref']]
        tab = to_check.copy()
        for lien in ['partner', 'pere', 'mere']:
            tab = tab.merge(to_check, left_on=lien, right_on='id', suffixes=('', '_' + lien), how='left', sort=False)
        tab.index = tab['id']

        for parent in ['pere', 'mere']:
            diff_age_pere = (tab['age_en_mois_' + parent] - tab['age_en_mois'])
            cond = diff_age_pere <= 12 * 14
            log.info(u"on retire " + str(sum(cond)) + " lien enfant " + parent + u" car l'âge n'était pas le bon")
            individus.loc[cond, parent] = -1

            cond = (tab['partner'] > -1) & (tab[parent] > -1) & \
                (tab[parent] == tab[parent + '_partner']) & \
                (tab['idmen'] != tab['idmen_' + parent])
            log.info("on retire " + str(sum(cond)) + " lien enfant " + parent + " car le partner a le même parent")
            individus.loc[(cond[cond]).index, parent] = -1

        self._check_links(individus)
        self.entity_by_name['individus'] = minimal_dtype(individus)
        all = self.entity_by_name['menages'].columns.tolist()
        enfants_hdom = [x for x in all if x[:3] == 'hod']
        self.drop_variable({
            'individus': ['enf', 'per1e', 'mer1e', 'grandpar'] + ['jepnais', 'jemnais', 'jemprof'],
            'menages': enfants_hdom
            })

    def match_couple_hdom(self):
        u'''
        Certaines personnes se déclarent en couple avec quelqu'un ne vivant pas au domicile, on les reconstruit ici.
        Cette étape peut s'assimiler à de la fermeture de l'échantillon.
        On sélectionne les individus qui se déclarent en couple avec quelqu'un hors du domicile.
        On match mariés,pacsé d'un côté et sans contrat de l'autre. Dit autrement, si on ne trouve pas de partenaire à
        une personne mariée ou pacsé on change son statut de couple.
        Comme pour les liens parents-enfants, on néglige ici la possibilité que le partner soit hors champ (étrange,
        prison, casernes, etc).
        Calcul aussi la variable individus['nb_enf']
        '''
        log.info(u"Début du matching des couples hors du domicile")
        individus = self.entity_by_name['individus']
        couple_hdom = individus['couple'] == 2
        # vu leur nombre, on regroupe pacsés et mariés dans le même sac
        individus.loc[(couple_hdom) & (individus['civilstate'] == 5), 'civilstate'] = 1
        # note que du coup, on cherche un partenaire de pacs parmi le sexe opposé. Il y a une petite par technique là
        # dedans qui fait qu'on ne gère pas les couples homosexuels

        # nb d'enfant
        individus.index = individus['id']
        nb_enf_mere = DataFrame(individus.groupby('mere').size(), columns = ['nb_enf'])
        nb_enf_mere['id'] = nb_enf_mere.index.values
        nb_enf_pere = DataFrame(individus.groupby('pere').size(), columns = ['nb_enf'])
        nb_enf_pere['id'] = nb_enf_pere.index
        # On assemble le nombre d'enfants pour les peres et meres en enlevant les manquantes ( = -1)
        enf_tot = nb_enf_mere[nb_enf_mere['id'] != -1].append(nb_enf_pere[nb_enf_pere['id'] != -1]).astype(int)
        individus['nb_enf'] = 0
        individus.loc[enf_tot['id'].values, 'nb_enf'] = enf_tot['nb_enf']

        men_contrat = couple_hdom & (individus['civilstate'].isin([1, 5])) & (individus['sexe'] == 0)
        women_contrat = couple_hdom & (individus['civilstate'].isin([1, 5])) & (individus['sexe'] == 1)
        men_libre = couple_hdom & (~individus['civilstate'].isin([1, 5])) & (individus['sexe'] == 0)
        women_libre = couple_hdom & (~individus['civilstate'].isin([1, 5])) & (individus['sexe'] == 1)

        individus['age'] = individus['age_en_mois'] // 12
        var_match = ['age', 'findet', 'nb_enf']  # ,'classif', 'dip6'
        score = "- 0.4893 * other.age + 0.0131 * other.age **2 - 0.0001 * other.age **3 "\
            " + 0.0467 * (other.age - age)  - 0.0189 * (other.age - age) **2 + 0.0003 * (other.age - age) **3 " \
            " + 0.05   * (other.findet - findet) - 0.5 * (other.nb_enf - nb_enf) **2 "
        match_contrat = Matching(individus.loc[women_contrat, var_match], individus.loc[men_contrat, var_match], score)
        match_found = match_contrat.evaluate(orderby=None, method='cells')
        individus.loc[match_found.values, 'partner'] = match_found.index
        individus.loc[match_found.index, 'partner'] = match_found.values

        match_libre = Matching(individus.loc[women_libre, var_match], individus.loc[men_libre, var_match], score)
        match_found = match_libre.evaluate(orderby=None, method='cells')
        individus.loc[match_found.values, 'partner'] = match_found.index
        individus.loc[match_found.index, 'partner'] = match_found.values

        # TODO: on pourrait faire un match avec les restants
        # au lieu de ça, on les considère célibataire
        individus.loc[men_contrat & (individus['partner'] == -1), ['civilstate', 'couple']] = [2, 3]
        individus.loc[women_contrat & (individus['partner'] == -1), ['civilstate', 'couple']] = [2, 3]
        individus.loc[men_libre & individus['partner'] == -1, ['civilstate', 'couple']] = [2, 3]
        individus.loc[women_libre & individus['partner'] == -1, ['civilstate', 'couple']] = [2, 3]

        individus.drop(['couple', 'age'], axis = 1, inplace = True)

        # Absence de polygames (seul -1 est partner de plusieurs individus)
        assert list(individus.partner.value_counts()[individus.partner.value_counts() > 1].index) == [-1]
        self.entity_by_name['individus'] = individus

    def calmar_demography(self):
        # TODO: Cette méthode doit être utilisée avec précaution. Elle a été introduite
        # pour recaler des données ne contenant pas les individus en institution
        individus = self.entity_by_name['individus']
        menages = self.entity_by_name['menages']
        individus_extraction = individus[['age', 'idmen', 'sexe']].copy()
        menages_extraction = menages[['id', 'pond']].copy()
        decades_by_sexe = dict()

        for sexe in [0, 1]:
            individus_extraction['decade'] = individus_extraction.age.loc[individus_extraction.sexe == sexe] // 10
            decades = individus_extraction.decade.loc[individus_extraction.sexe == sexe].unique()
            for decade in list(decades):
                individus_extraction['dummy_decade'] = (individus_extraction.decade == decade) * 1
                dummy = individus_extraction[['dummy_decade', 'idmen']].groupby(by = 'idmen').sum().reset_index()
                dummy.rename(
                    columns = {'dummy_decade': '{}_{}'.format(str(int(decade)), sexe), 'idmen': 'id'}, inplace = True)
                menages_extraction = menages_extraction.merge(dummy, on = 'id')
            decades_by_sexe[sexe] = ['{}_{}'.format(str(int(decade)), sexe) for decade in decades]
        data_in = dict()

        for col in decades_by_sexe[0] + decades_by_sexe[1] + ['id', 'pond']:
            data_in[col] = menages_extraction[col].values

        margins_by_decade = dict()
        for sexe in ['male', 'female']:
            insee = get_data_frame_insee(sexe)[2010]
            sexe_number = 0 if sexe == 'male' else 1
            insee.index = ['{}_{}'.format(decade, sexe_number) for decade in insee.index]
            margins_by_decade.update(insee.to_dict())

        log.info('''Marges par décade et sexe.
Hommes: {}
Femmes: {}'''.format(
                [(decade, margins_by_decade['{}_0'.format(decade)]) for decade in range(0, 11)],
                [(decade, margins_by_decade['{}_1'.format(decade)]) for decade in range(0, 11)]
                ))
        parameters = dict(
            method = 'logit',
            lo = 1.0 / 3.0,
            up = 3.0
            )
        pondfin_out, lambdasol, margins_new_dict = calmar(
            data_in, margins_by_decade, parameters = parameters, pondini = 'pond')

        # check_calmar(data_in, margins_by_decade, pondini = 'pond', pondfin_out = pondfin_out, lambdasol = lambdasol,
        #    margins_new_dict = margins_new_dict)

        menages['pondini'] = menages.pond.copy()
        menages.pond = pondfin_out
        self.entity_by_name['menages'] = menages


def test_build():
    logging.basicConfig(level = logging.INFO, stream = sys.stdout)
    import time
    start_t = time.time()
    data = Patrimoine()
    data.load()
    # drop_variable() doit tourner avant table_initial() car on fait comme si diplome par exemple n'existait pas
    # plus généralement, on aurait un problème avec les variables qui sont renommées.
    data.to_DataTil_format()
    data.champ()
    data.calmar_demography()
    # data.work_on_past() TODO: à réactiver !
    # data.create_past_table()
    data.drop_variable()
    # data.corrections()
    data.partner()
    data.enfants()
    data.expand_data(weight_threshold = 200)

    data.creation_child_out_of_house()
    data.matching_par_enf()
    data.match_couple_hdom()
    data.creation_foy()
    data.format_to_liam()
    data.final_check()
    data.store_to_liam()
    menages = data.entity_by_name['menages']
    individus = data.entity_by_name['individus']
    individus_institutions = data.entity_by_name['individus_institutions']

    log.info(u"Temps de calcul : {} s".format(time.time() - start_t))
    log.info(u"Nombre d'individus de la table final : {}".format(len(individus)))

    # des petites verifs finales
    individus['en_couple'] = individus['partner'] > -1
    test = individus['partner'] > -1

    assert list(individus.partner.value_counts()[individus.partner.value_counts() > 1].index) == [-1]
    log.info(individus.groupby(['civilstate', 'en_couple']).size())
    assert individus.query('civilstate == 1')['en_couple'].all()
    assert individus.query('civilstate == 5')['en_couple'].all()
    log.info(individus['data_origin'].value_counts(dropna = False))
    log.info(individus_institutions['data_origin'].value_counts(dropna = False))


if __name__ == '__main__':
    test_build()
