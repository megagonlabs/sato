import numpy  as np
import pandas as pd
import os
from scipy       import stats
from collections import OrderedDict
SHERLOCKPATH = os.environ['SHERLOCKPATH']

embedding_loc = os.path.join(SHERLOCKPATH, 'pretrained', 'glove.6B')

word_vectors_f = open(os.path.join(embedding_loc ,'glove.6B.50d.txt'))
word_to_embedding = {}
for l in word_vectors_f:
    term, vector = l.strip().split(' ', 1)
    vector = np.array(vector.split(' '), dtype=float)
    word_to_embedding[term] = vector

num_embeddings = 50
# Input: a single column in the form of a pandas series
def extract_word_embeddings_features(values):
    
    f = OrderedDict()
    embeddings = []
    
    values = values.dropna()

    for v in values:
        v = str(v).lower()
        if v in word_to_embedding:
            embeddings.append(word_to_embedding.get(v))
        else:
            words = v.split(' ')
            embeddings_to_all_words = []

            for w in words:
                if w in word_to_embedding:
                    embeddings_to_all_words.append(word_to_embedding.get(w))
            if embeddings_to_all_words:
                mean_of_word_embeddings = np.nanmean(embeddings_to_all_words, axis=0)
                embeddings.append(mean_of_word_embeddings)

                
    if len(embeddings) == 0: 
        
        for i in range(num_embeddings): f['word_embedding_avg_{}'.format(i)]  = np.nan
        for i in range(num_embeddings): f['word_embedding_std_{}'.format(i)]  = np.nan
        for i in range(num_embeddings): f['word_embedding_med_{}'.format(i)]  = np.nan
        for i in range(num_embeddings): f['word_embedding_mode_{}'.format(i)] = np.nan
        
        #f['dummy'] = 0
        return f
    
    else:
    
        mean_embeddings = np.nanmean(embeddings, axis=0)
        med_embeddings  = np.nanmedian(embeddings, axis=0)
        std_embeddings  = np.nanstd(embeddings, axis=0)
        mode_embeddings = stats.mode(embeddings, axis=0, nan_policy='omit')[0].flatten()

        for i, e in enumerate(mean_embeddings):
            f['word_embedding_avg_{}'.format(i)] = e
            
        for i, e in enumerate(std_embeddings):
            f['word_embedding_std_{}'.format(i)] = e

        for i, e in enumerate(med_embeddings):
            f['word_embedding_med_{}'.format(i)] = e
        
        for i, e in enumerate(mode_embeddings):
            f['word_embedding_mode_{}'.format(i)] = e
        
        #f['dummy'] = 1
        
        return f