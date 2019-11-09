import numpy  as np
import pandas as pd
import random
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import OrderedDict

SHERLOCKPATH = os.environ['SHERLOCKPATH']
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Input: a collection of columns stored in a dataframe column 'values'
# Only needed for training.
def tagcol_paragraph_embeddings_features(train_data):
    
    # Expects a dataframe with a 'values' column
    train_data_values = train_data['values']
    columns = [TaggedDocument( random.sample(col, min(1000, len(col))) , [i]) for i, col in enumerate(train_data_values.values)]
    
    return columns

# Input: returned tagged document collection from tagcol_paragraph_embeddings_features
# Only needed for training.
def train_paragraph_embeddings_features(columns, dim):
        
    print('Training paragraph vectors with vector dimension: ', dim)
    
    # TRAN PARAGRAPH VECTORS MODEL
    model = Doc2Vec(columns, dm=0, negative=3, workers=8, vector_size=dim, epochs=20, min_count=2, seed=13)
    
    model_file = os.path.join(SHERLOCKPATH, 'pretrained', 'par_vec_trained_{}.pkl'.format(dim))
    model.save(model_file)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    

# Input: a single column in the form of a pandas Series.
def infer_paragraph_embeddings_features(data, dim):
    
    vec = data 
    embeddings = []
    
    # INFER PARAGRAPH VECTORS
    model = Doc2Vec.load(os.path.join(SHERLOCKPATH, 'pretrained', 'par_vec_trained_{}.pkl'.format(dim))) 

    f = OrderedDict()
    
    if len(vec) > 1000:
        vec = random.sample(vec, 1000)
        
    vec = pd.Series(model.infer_vector(vec, steps=20, alpha=0.025))
    
    
    for i in range(len(vec)):
        f['par_vec_{}'.format(i)] = vec[i]

    return f