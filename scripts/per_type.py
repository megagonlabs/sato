from analysis_functions import data_gen_all
import os
basepath = os.environ['BASEPATH']

path = os.path.join(basepath, 'results/CRF_log/type78/CRF_path/outputs')
pathL = os.path.join(basepath, 'results/CRF_log/type78/CRF+LDA_pathL/outputs')

path_multi_col = os.path.join(basepath, 'results/CRF_log/type78/CRF_path_multi-col/outputs')
pathL_multi_col = os.path.join(basepath, 'results/CRF_log/type78/CRF+LDA_pathL_multi-col/outputs')


data_gen_all(path, pathL, 'all-tables', './output')
data_gen_all(path_multi_col, pathL_multi_col, 'multi-col', './output')

