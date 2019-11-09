'''
Module used to extract VizML features from a specificd corpus,
Only extract columns that has a valid header from ./headers/{}_header_valid.csv
'''
import os
import sys
from os.path import join
import argparse
import pandas as pd
import time
import itertools
from multiprocessing import Pool
from tqdm import tqdm
import functools

from helpers.read_raw_data import get_filtered_dfs_by_corpus
from utils import get_valid_types, str_or_none, str2bool
from helpers.utils import valid_header_iter_gen, count_length_gen
# Get rid of gensim deprecated warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)
valid_header_dir = os.path.join(os.environ['BASEPATH'], 'extract', 'out', 'headers', TYPENAME)


if __name__ == "__main__": 


    MAX_FIELDS = 10000
    cache_size = 100

    # Get corpus
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_chunk', type=str)
    parser.add_argument('-f', '--features', type=str, nargs='?', default='sherlock', choices=['sherlock', 'topic'])
    parser.add_argument('-LDA', '--LDA_name', nargs='?', type=str_or_none, default=None)
    parser.add_argument('-n', '--num_processes', nargs='?', type=int, default=4)
    parser.add_argument('-o', '--overwrite', nargs='?', type=str2bool, default=False)

    args = parser.parse_args()
    corpus = args.corpus_chunk

    # Create features directory
    features_dir = join(os.environ['BASEPATH'], 'extract', 'out', 'features', TYPENAME)
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)

    if args.features == 'topic':
        assert args.LDA_name is not None, "Must provide an LDA_name"

        os.environ['LDA_name'] = args.LDA_name
        # hack to pass in LDA name for extractor
        from feature_extraction.topic_features_LDA import extract_topic_features
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.ldamodel import LdaModel

        feature_name = "{}-{}".format(args.features, args.LDA_name)
        extract_func = extract_topic_features

    elif args.features == 'sherlock':
        from feature_extraction.sherlock_features import extract_sherlock_features
        feature_name = args.features
        extract_func = extract_sherlock_features
    else:
        print('Invalid feature names')
        exit(1)


    print('Extracting features for corpus {}, feature group {}'.format(corpus, args.features))

    output_file = join(features_dir, '{}_{}_{}_features.csv'.format(corpus, TYPENAME, feature_name))
    if not args.overwrite:
        assert not os.path.isfile(output_file), "\n {} already exists".format(output_file)

    header_name =  "{}_{}_header_valid.csv".format(corpus, TYPENAME)
    header_iter = valid_header_iter_gen(header_name)
    


 
    if corpus.startswith('webtables'):
        wcorpus, partition = corpus.split('-')
        raw_df_iter = get_filtered_dfs_by_corpus['webtables'](wcorpus, header_iter)
    else:
        if '-' in corpus:
            corpus= corpus.split('-')[0]
        raw_df_iter = get_filtered_dfs_by_corpus[corpus](header_iter)



    header_length = count_length_gen(os.path.join(valid_header_dir, header_name)) - 1
    print("Header Length", header_length) 

    ########################################
    # Distribute the tasks using pools
    ########################################
    task_pool = Pool(args.num_processes)
    

    counter = 0
    header, mode = True, 'w'
    col_counter = 0
    cache = []
    for df_features in tqdm(task_pool.imap(extract_func, raw_df_iter), 
                            total=header_length,
                            desc='{} processes'.format(args.num_processes)):
        counter += 1
        cache.append(df_features)
        if counter % cache_size == 0:
            df = pd.concat(cache)
            df.to_csv(output_file, header=header, index=False, mode=mode)
            col_counter += len(df)
            header, mode = False, 'a'
            cache = []

    #save the last cache
    if len(cache) > 0:
        df = pd.concat(cache)
        df.to_csv(output_file, header=header, index=False, mode=mode)

    print("Number of columns: {}".format(col_counter))

    task_pool.close()
    task_pool.join()
