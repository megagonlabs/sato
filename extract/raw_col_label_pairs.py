import os
from os.path import join
import pandas as pd
import json
import re
import argparse

##############################
# generate bert_input
##############################

# temp hack of setting path
#os.environ['BASEPATH'] = '/home/dan_z/col2type'

path = join(os.environ['BASEPATH'], 'extract', 'out', 'extracted_tables')
TYPENAME = os.environ['TYPENAME']

MAX_COL_LEN = 500
from helpers.read_raw_data import get_filtered_dfs_by_corpus


with open(os.path.join(os.environ['BASEPATH'], 'configs','types.json'), 'r') as typefile:  
    valid_types = json.load(typefile)[TYPENAME]


def canonical_header(h, max_header_len=30):
    # convert any header to its canonincal form
    # e.g. fileSize
    h = str(h)
    if len(h)> max_header_len:
        return '-'
    h = re.sub(r'\([^)]*\)', '', h) # trim content in parentheses
    h = re.sub(r"([A-Z][a-z])", r" \1", h) #insert a space before any Cpital starts
    words = list(filter(lambda x: len(x)>0, map(lambda x: x.lower(), re.split('\W', h))))
    if len(words)<=0:
        return '-'
    new_phrase = ''.join([words[0]] + [x.capitalize() for x in words[1:]])
    return new_phrase

def single_col(df):
    for i, c in enumerate(df.columns):
        yield c, ' '.join(df.iloc[:, i])


def with_surrounding(df):
    '''
    build input for bert 
    Input: [colA, samples from (colB, colC, colD)] => type of colA
    first 64 from colA + 64 from head of all other columns 

    '''
    col_count = len(df.columns)
    # sinlge column tables
    if col_count==1:
        yield df.columns[0], ' '.join(['None']*64) + ' ' + ' '.join(df.iloc[:64, 0])
    else:
        sample_count = int(64/(col_count-1))

        # prepare context info for each col
        content = []
        for idx, c in enumerate(df.columns):
            # taking only the first 64/n-1 words
            # cells could contain more than one word
            temp = ' '.join(df.iloc[:sample_count, idx]).split(' ')[:sample_count]
            content.append(' '.join(temp))

        for idx, c in enumerate(df.columns):
            other_cols = ' '.join(content[:idx] + content[idx+1: ])

            yield c, other_cols + ' ' + ' '.join(df.iloc[:64, idx])
    

def process_table(df, process_type=None):
    # Return only columns with valid headers
    # in the form of (column_values, label)

    # convert headers to its canonical form
    df = df.astype('str', copy=False)
    df.rename(mapper=canonical_header, axis='columns', inplace=True)

    to_drop =[]
    for col in df.columns:
        if not col in valid_types:
            to_drop.append(col)

    df = df.drop(to_drop, axis=1)

    if process_type=='single':
        yield from single_col(df)
    elif process_type=='sur':
        yield from with_surrounding(df)
    else:
        print('Invalid process_type')
        exit(1)



def table_gen(corpus, source, process_type=None):

    if source=='headers':
        if corpus.startswith('webtables'):
            wcorpus, partition = corpus.split('-')
            df_iter = get_filtered_dfs_by_corpus['webtables'](wcorpus,partition)
        else:
            df_iter = get_filtered_dfs_by_corpus[corpus]()

        for d in df_iter:
            df = d['df']
            if len(df) > MAX_COL_LEN:
                df = df.sample(n=MAX_COL_LEN)

            yield from process_table(df, process_type)

    elif source =='tables':

        file_list = list(filter(lambda x: not x.startswith('.'), os.listdir(join(path, corpus))))
        print("Number of tables in {} corpus: {}".format(corpus, len(file_list)))

        for f in file_list:
            #print(f)
            df = pd.read_csv(join(path, corpus, f),
                             index_col=0, 
                             error_bad_lines=False,
                             nrows=MAX_COL_LEN)

            if len(df) > MAX_COL_LEN:
                df = df.sample(n=MAX_COL_LEN)   

            yield from process_table(df, process_type)


    else:
        print("Invalid source")
        exit(1)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',  nargs='?', type=str,
                    help='Corpus name')
    parser.add_argument('--source', nargs='?', type=str, choices=['headers','tables'], default='headers',
                    help='Generate from header file or extracted tables.')

    parser.add_argument('--p_type', nargs='?', type=str, choices=['single','sur'], default='single',
                    help='Generate single column content or with sample from other columns')

    parser.add_argument('--col_limit', type=int, nargs='?', default=-1, help='max number of columns')

    args = parser.parse_args()


    out_path = join(os.environ['BASEPATH'], 'extract', 'out', 'bert_input')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_name = join(out_path, "{}_{}.csv".format(args.corpus,args.p_type))
    if os.path.isfile(out_name):
        print("\n \n {} already exists".format(out_name))
        exit(1)

    tables = table_gen(args.corpus, args.source,args.p_type)

    chunk_size = 500
    col_count = 0

    # write header
    df = pd.DataFrame(columns=['label', 'content'])
    df.to_csv(out_name, index = False, header=True)

    chunk_cache = []

    for t in tables:
        chunk_cache.append(t)

        if args.col_limit > 0 and col_count >= args.col_limit-1:
            break
        if col_count % chunk_size == 0 and col_count > 0:
            df = pd.DataFrame(chunk_cache)
            df.to_csv(out_name, mode='a', index=False, header=False)
            chunk_cache = []

            print("Columns processed: ", col_count)
        col_count +=1

    df = pd.DataFrame(chunk_cache)
    df.to_csv(out_name, mode='a', index=False, header=False)

