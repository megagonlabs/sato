import string
import numpy  as np
from scipy.stats import skew, kurtosis
from collections import OrderedDict


# Input: a single column in the form of pandas series
def extract_bag_of_characters_features(data, n_val):
    
    characters_to_check = [ '['+  c + ']' for c in string.printable if c not in ( '\n', '\\', '\v', '\r', '\t', '^' )] + ['[\\\\]', '[\^]']
    
    f = OrderedDict()

    f['n_values'] = n_val
    data_no_null = data.dropna()
    all_value_features = OrderedDict()

    all_value_features['length'] = data_no_null.apply(len)

    for c in characters_to_check:
        all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)
        
    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    n_none = data.size - data_no_null.size - len([ e for e in data if e == ''])
    f['none-agg-has'] = n_none > 0
    f['none-agg-percent'] = n_none / len(data)
    f['none-agg-num'] = n_none
    f['none-agg-all'] = (n_none == len(data))
    #print(len(f))
    return f


    
    

