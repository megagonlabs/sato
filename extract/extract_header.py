'''
Extract headers from a corpus, 
Filter header before generating feature vectors
Format of _header_valid.csv:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Columns:        Descriptions
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Locator         The path of the datafile
dataset_id      The dataset of the table. one file has multiple tables except manyeyes
field_list      The list of field whose header is a valid type. numbers are index in table
field_names     The list of field names. numbers are index in the corresponsing type list

'''

import os
import sys
from os.path import join
import argparse
import json
import pandas as pd
from collections import OrderedDict
import itertools
from helpers.utils import canonical_header, long_name_digest
from tqdm import tqdm
from utils import get_valid_types, str_or_none, str2bool

from helpers.read_raw_data import get_dfs_by_corpus

# Get rid of gensim deprecated warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#TODO (Dan): update MAX_FIELD
MAX_FIELDS = 10000
MAX_HEADER_LEN = 30
cache_size = 10
TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)



def get_valid_headers(df_iter):
    for df_dic in df_iter:
        df, locator, dataset_id = df_dic['df'], df_dic['locator'], df_dic['dataset_id']

        if(df.shape[1] > MAX_FIELDS):
            print('Exceeds max fields')
            continue

        valid_fields = []
        field_names = [] # index of the headers, according to the type used.
        for field_order, field_name in enumerate(df.columns):

            # canonicalize headers
            field_name_c = canonical_header(field_name)
            
            # filter the column name
            if field_name_c in valid_types:
                valid_fields.append(field_order)
                field_names.append(valid_types.index(field_name_c))
                

        if len(valid_fields) > 0:
            table_valid_headers = OrderedDict()

            table_valid_headers['locator'] = locator
            table_valid_headers['dataset_id'] = dataset_id
            table_valid_headers['field_list'] = valid_fields
            table_valid_headers['field_names'] = field_names
            yield table_valid_headers
        



# Get corpus
parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=str)
parser.add_argument('--file_size', type=int, nargs='?', default=40000, help='number of tables in a file')
parser.add_argument('--file_number', type=int, nargs='?', default=3, help='number of files to generate')
parser.add_argument('-o', '--overwrite', nargs='?', type=str2bool, default=False)


args = parser.parse_args()
corpus = args.corpus


# Create features directory
header_path = join(os.environ['BASEPATH'], 'extract', 'out', 'headers', TYPENAME)
if not os.path.exists(header_path):
    os.mkdir(header_path)



print('Extracting headers for corpus:{}, TYPENAME: {}'.format(corpus, TYPENAME))
batch_tables = []
if corpus.startswith('webtables'):
    df_iter = get_dfs_by_corpus['webtables'](corpus)
else:
    df_iter = get_dfs_by_corpus[corpus]()

header_iter = get_valid_headers(df_iter)

#TODO(Dan): better skipping functionality
start = 0
header_iter = itertools.islice(header_iter, start, None, None)
print("Start from number {}".format(start))

for f_idx in range(args.file_number):
    
    header_name = '{}-p{}_{}_header_valid.csv'.format(corpus, f_idx+1, TYPENAME)
    header_file_name = join(header_path, header_name)
    if not args.overwrite:
        assert not os.path.isfile(header_file_name), "\n \n {} already exists".format(header_file_name)


    counter = 0
    header, mode = True, 'w'
    col_counter = 0
    cache = []
    for table_header in tqdm(itertools.islice(header_iter, 0, args.file_size, None),
                             total=args.file_size,
                             desc=header_name):
        counter += 1
        cache.append(table_header)
        if counter % cache_size == 0:
            df = pd.DataFrame(cache)
            df.to_csv(header_file_name, header=header, index=False, mode=mode)
            col_counter += len(df)
            header, mode = False, 'a'
            cache = []

    #save the last cache
    df = pd.DataFrame(cache)
    df.to_csv(header_file_name, header=header, index=False, mode=mode)


