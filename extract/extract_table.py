'''
Save the extracted tables to file by iterating through the valid header file
'''
import os
import sys
from os.path import join
import argparse
import math
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import itertools
from helpers.utils import long_name_digest, valid_header_iter_gen, count_length_gen
from helpers.read_raw_data import get_filtered_dfs_by_corpus


TYPENAME = os.environ['TYPENAME']
valid_header_dir = os.path.join(os.environ['BASEPATH'], 'extract', 'out', 'headers', TYPENAME)


extrated_table_path = join(os.environ['EXTRACTPATH'], 'out', 'extracted_tables', TYPENAME)
sample_every_N = 1000

# Get corpus
parser = argparse.ArgumentParser()
parser.add_argument('corpus_chunk', type=str)
parser.add_argument('-sp', '--sample', nargs='?', type=int, default=-1,  help='sample every * rows')
#parser.add_argument('-t', '--terminate', nargs='?', type=int, default=-1, help='terminate after * rows')
parser.add_argument('-n', '--num_processes', nargs='?', type=int, default=4)

args = parser.parse_args()
corpus = args.corpus_chunk

# Create output directory
table_loc = join(extrated_table_path, corpus)
print(table_loc)
if not os.path.exists(table_loc):
    os.makedirs(table_loc)


header_name =  "{}_{}_header_valid.csv".format(corpus, TYPENAME)
header_iter = valid_header_iter_gen(header_name)
header_length = count_length_gen(os.path.join(valid_header_dir, header_name)) - 1
#print("Header Length", header_length)

if corpus.startswith('webtables'):
    wcorpus, partition = corpus.split('-')
    raw_df_iter = get_filtered_dfs_by_corpus['webtables'](wcorpus, header_iter)
else:
    if '-' in corpus:
        corpus= corpus.split('-')[0]
    raw_df_iter = get_filtered_dfs_by_corpus[corpus](header_iter)



def save_file(df_dic):
    df, locator, dataset_id = df_dic['df'], df_dic['locator'], df_dic['dataset_id']

    id_digest = long_name_digest(dataset_id)
    save_name = '{}_{}_{}.csv'.format(locator.replace('/',';'), dataset_id.replace('/',';')[:30], id_digest)

    df.to_csv(join(table_loc, save_name))

    return


if args.sample != -1:
    total_file_num = math.ceil(header_length/float(args.sample))
    df_iter = itertools.islice(raw_df_iter, None, None, args.sample)
else:
    total_file_num = header_length 
    df_iter = raw_df_iter

########################################
# Distribute the tasks using pools
########################################
task_pool = Pool(args.num_processes)
counter = 0

print('Extracting tables from headers for corpus:{}. Saving to {}'.format(corpus, extrated_table_path))

for _ in tqdm(task_pool.imap(save_file, df_iter), 
                        total=total_file_num,
                        desc='{} processes'.format(args.num_processes)):
    counter += 1


task_pool.close()
task_pool.join()

print("Number of files saved: {}".format(counter))



