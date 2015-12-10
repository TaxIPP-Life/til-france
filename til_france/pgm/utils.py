# -*- coding:utf-8 -*-

'''
Created on 5 août 2013
@author: a.eidelman
'''
from numpy import array, arange, ones
from numpy.lib.stride_tricks import as_strided
from pandas import Series, DataFrame

of_name_to_til = {'individus':'individus', 'foyers_fiscaux':'foyers_fiscaux', 'menages':'menages', 'familles':'familles'}
til_name_to_of  = dict ( (v,k) for k, v in of_name_to_til.items() )


def concatenated_ranges(ranges_list) :
    ranges_list = array(ranges_list, copy=False)
    base_range = arange(ranges_list.max())
    base_range = as_strided(base_range,
                             shape=ranges_list.shape + base_range.shape,
                             strides=(0,) + base_range.strides)
    return base_range[base_range < ranges_list[:, None]]


def output_til_to_liam(output_til, index_til, context_id):
    ''' pour jouer avec les indices mais est inutile maintenant je pense '''
    output_liam = Series(- ones(len(context_id)), index=context_id)
    if isinstance(output_til, DataFrame) or isinstance(output_til, Series):
        output_liam[output_til.index.values] = output_til.values
        return array(output_liam)
    else:
        output_liam[index_til] = output_til
        return array(output_liam)
