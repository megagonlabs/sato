import pandas as pd
import os 
from os.path import join
import json
import numpy as np
import math


from utils import load_tmp_df, get_valid_types

TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)
L = len(valid_types)


tmp_path = join(os.environ['BASEPATH'], 'tmp')
path = join(os.environ['BASEPATH'],'extract', 'out', 'headers', TYPENAME)



# distribution of types

corpus_list = ['webtables0-p1', 'webtables0-p1', 'webtables0-p3']
matrix_co = np.zeros((L,L))
matrix_co_full = np.zeros((L,L))

type_counts = {}
table_counts = 0
column_count_list = []

for corpus in corpus_list:
    df = load_tmp_df(path, tmp_path, "{}_{}_header_valid".format(corpus, TYPENAME), table=True) 
    for idx, row in df.iterrows():
        
        fields = sorted(list(eval(row['field_names'])))
        # update stats
#        column_count_list.append(len(fields))
#        table_counts += 1 if len(fields)>1 else 0
#        for t_idx in fields:
#            t = valid_types[t_idx]
#            type_counts[t]= type_counts.get(t, 0)+1
        # update co-occurrence matrix
        for idx_l, l in enumerate(fields):
            for idx_r in range(idx_l+1, len(fields)):
                matrix_co[l, fields[idx_r]]+=1
                matrix_co_full[l, fields[idx_r]]+=1
                matrix_co_full[fields[idx_r], l]+=1
                
print(matrix_co_full.shape)
np.save('matrix_co_W0_{}.npy'.format(TYPENAME), matrix_co_full)