# -*- coding: utf-8 -*-

import datetime as dt
import pandas as pd

try:
    from til_pension.pension_data import PensionData
    from til_pension.pension_legislation import PensionParam, PensionLegislation
    from til_pension.simulation import PensionSimulation
except ImportError:
    til_pension = None


def get_pension(context, yearleg):
    ''' return a PensionSimulation '''
    salaire_imposable = context['longitudinal']['salaire_imposable']
    workstate = context['longitudinal']['workstate']
    # calcul de la date de naissance au bon format
    datesim = context['period']
    datesim_in_month = 12 * (datesim // 100) + datesim % 100
    datenaiss_in_month = datesim_in_month - context['age_en_mois']
    naiss = 100 * (datenaiss_in_month // 12) + datenaiss_in_month % 12 + 1
    naiss = pd.Series(naiss)
    naiss = pd.Series(naiss).map(lambda t: dt.date(t // 100, t % 100, 1))

    info_ind = pd.DataFrame({
        'index': context['id'], 'age_en_mois': context['age_en_mois'], 'naiss': naiss, 'sexe': context['sexe'],
        'nb_enf_all': context['nb_enf'], 'nb_pac': context['nb_pac'], 'nb_enf_RG': context['nb_enf_RG'],
        'nb_enf_RSI': context['nb_enf_RSI'], 'nb_enf_FP': context['nb_enf_FP'], 'tauxprime': context['tauxprime']
        })
    info_ind = info_ind.to_records(index=False)

    workstate = workstate.loc[workstate['id'].isin(info_ind.index), :].copy()
    workstate.set_index('id', inplace=True)
    workstate.sort_values(inplace=True)
    salaire_imposable = salaire_imposable.loc[salaire_imposable['id'].isin(info_ind.index), :].copy()
    salaire_imposable.set_index('id', inplace=True)
    salaire_imposable.sort_values(inplace=True)
    salaire_imposable.fillna(0, inplace=True)

    data = PensionData.from_arrays(workstate, salaire_imposable, info_ind)
    param = PensionParam(yearleg, data)
    legislation = PensionLegislation(param)
    simulation = PensionSimulation(data, legislation)
    simulation.set_config()
    return simulation
