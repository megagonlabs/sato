import os
import sys
from os.path import join
import argparse
import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import math
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from utils import dic2name

TYPENAME = os.environ['TYPENAME']
LDA_CACHE = join('LDA_cache', TYPENAME)

def clean(s):
    tokens = nltk.word_tokenize(s.lower())
    tokens_clean = [token for token in tokens if token not in stopwords.words('english')]
    tokens_stemmed = [PorterStemmer().stem(token) for token in tokens_clean]
    return tokens_stemmed


def tokenize(col, **kwargs):
    threshold = int(kwargs['thr'])
    ret = []
    for st in col:
        if len(st)> threshold:
            # tokenize the string if longer than threshold
            # and append a longstr tag
            ret.extend(clean(st))
            if threshold > 0:
                ret.append('longstr')
        else:
            ret.append(st.lower())
    return ret


def process_col(col, **kwargs):

    numeric = kwargs['num']
    # process the cols to return a bags of word representation
    if col.dtype == 'int64' or col.dtype =='float64':
        if numeric == 'directstr':
            return list(col.astype(str))
        elif numeric == 'placeholder':
            return [str(col.dtype)] * len(col)
        
    if col.dtype == 'object':
        return tokenize(list(col.astype(str)), **kwargs)
    
    else:
        return list(col.astype(str))
       
    return col

def corpus_iter(base_path, paths, batch_size, limit, **kwargs):
    # iterate through tables in a directory, collect all values in each table.
    f_list = []
    for t in table_paths:
        #temp_list = list(filter(lambda x: x.endswith('.csv'), os.listdir(join(base_path, t))))
        temp_list = list(filter(lambda x: not x.startswith('.'), os.listdir(join(base_path, t))))

        temp_list = [join(t, f) for f in temp_list]
        f_list.extend(temp_list)

    if limit>0:
        f_list = f_list[:limit]    
    l = len(f_list)
    print("total # of tables", l)

    for b in range(math.ceil(l/float(batch_size))):    

        corpus = []
        for f in f_list[b*batch_size: min((b+1)*batch_size, l)]:
            try:
                df = pd.read_csv(join(base_path,f),
                                 index_col=0, 
                                 error_bad_lines=False)#.astype(str) 
            except Exception as e:
                print("Exception loading manyeyes data", e)
                continue
                
            table_seq = []
            for col in df.columns:
                processed_col = process_col(df[col], **kwargs)
                table_seq.extend(processed_col)
            corpus.append(table_seq)

        yield corpus
             

def train_LDA(base_path, table_paths, batch_size, limit, use_dictionary=False, **kwargs):

    model_name = dic2name(kwargs)
    print("Model: ", model_name)
    topic_num = kwargs['tn']

    # Pass 1 get the dictionary
    if use_dictionary=='True':
        dic = Dictionary.load(join(LDA_CACHE, 'dictionary_{}'.format(model_name)))
    else:

        dic = Dictionary([])
        b = 0
        for corpus in corpus_iter(base_path, table_paths, batch_size, limit, **kwargs):
            dic.add_documents(corpus)
            print('Dictionary batch {}: current dic size {}'.format(b, len(dic)))
            b+=1
            
        # save dictionary
        dic.save(join(LDA_CACHE, 'dictionary_{}'.format(model_name)))

    print("Dictionary size", len(dic))
    
    # Pass 2 train LDA
    whole_corpus = corpus_iter(base_path, table_paths, batch_size, limit, **kwargs)
    first_batch = next(whole_corpus)
    first_bow = [dic.doc2bow(text, allow_update=False) for text in first_batch]
    #print(first_bow)

    lda = LdaModel(first_bow, id2word=dic, num_topics=topic_num, minimum_probability=0.0)
    batch_no = 0
    print('LDA update batch {}'.format(batch_no))

    for batch in whole_corpus:
        batch_bow = [dic.doc2bow(text, allow_update=False) for text in batch]
        #print(corpus_bow)
        lda.update(batch_bow)
        batch_no +=1
        print('LDA update batch {}'.format(batch_no))
    
    # Save model to disk.
    temp_file = join(LDA_CACHE, "model_{}".format(model_name))
    lda.save(temp_file)
    
    print("Training from {} done. Batch_size: {}, long str tokenization threshold: {}, numerical representations: {}.\
          \nTotal size of dictionary: {}".format(table_paths, batch_size, kwargs['thr'], kwargs['num'], len(dic)))
    return


if __name__ == "__main__":

    if not os.path.exists(LDA_CACHE):
        os.makedirs(LDA_CACHE)

    # Get corpus
    parser = argparse.ArgumentParser()
    #parser.add_argument('table_loc', type=str, help='location of tables')
    parser.add_argument('table_loc', type=str, help='folder names of tables, could be multiple names, seperated by comma')
    parser.add_argument('--file_limit', type=int, nargs='?', default=-1, help='max number of files used')
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=5000)
    parser.add_argument('-l', '--long_threshold', nargs='?', type=int, default=0, help='Long strings will be tokenized and tag as longstr')
    parser.add_argument('-num', '--numeric_rep', nargs='?', type=str, default='directstr', choices=['directstr', 'placeholder', 'bin'],
                         help='Convert numbers to strings directly or replace with int64/float64')
    
    parser.add_argument('--topic_num', nargs='?', type=int, default=100, help='Number of topics')

    parser.add_argument( '--use_dictionary', nargs='?', type=str, default='False', choices=['True', 'False'],
                         help='if set to True, use existing dictionary in LDA_cache')
    args = parser.parse_args()


    base_path = join(os.environ['EXTRACTPATH'],'out', 'extracted_tables', TYPENAME)
    table_paths = args.table_loc.split(',')
    
    batch_size = args.batch_size

    kwargs = {'thr': args.long_threshold, 'num': args.numeric_rep, 'tn':args.topic_num}

    train_LDA(base_path, table_paths, batch_size, args.file_limit, args.use_dictionary, **kwargs)

