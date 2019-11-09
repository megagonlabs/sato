import math
import nltk
import numpy as np
from collections import OrderedDict


# Input: a single column in dthe form of a pandas series
def extract_bag_of_words_features(data):
    
    f = OrderedDict()
    data = data.dropna()
    
    n_val = data.size
    
    if not n_val: return
    
    # Entropy of column
    freq_dist = nltk.FreqDist(data)
    probs = [freq_dist.freq(l) for l in freq_dist]
    f['col_entropy'] = -sum(p * math.log(p,2) for p in probs)

    # Fraction of cells with unique content
    num_unique = data.nunique()
    f['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(data.str.contains('[0-9]', regex=True))
    text_cells = np.sum(data.str.contains('[a-z]|[A-Z]', regex=True))
    f['frac_numcells']  = num_cells / n_val
    f['frac_textcells'] = text_cells / n_val
    
    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    f['avg_num_cells'] = np.mean(data.str.count(num_reg))
    f['std_num_cells'] = np.std(data.str.count(num_reg))
    
    # Average + std number of textual tokens in cells
    text_reg = '[a-z]|[A-Z]'
    f['avg_text_cells'] = np.mean(data.str.count(text_reg))
    f['std_text_cells'] = np.std(data.str.count(text_reg))
    
    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    f['avg_spec_cells'] = np.mean(data.str.count(spec_reg))
    f['std_spec_cells'] = np.std(data.str.count(spec_reg))
    
    # Average number of words in each cell
    space_reg = '[" "]'
    f['avg_word_cells'] = np.mean(data.str.count(space_reg) + 1)
    f['std_word_cells'] = np.std(data.str.count(space_reg) + 1)

    
    return f

    
    
    

