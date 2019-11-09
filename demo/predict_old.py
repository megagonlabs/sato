import pandas as pd
import numpy as np
import random
import os
from features.bag_of_characters import extract_bag_of_characters_features
from features.bag_of_words      import extract_bag_of_words_features
from features.word_embeddings   import extract_word_embeddings_features
from features.paragraph_vectors import infer_paragraph_embeddings_features
import tensorflow

SHERLOCKPATH = os.environ['SHERLOCKPATH']

def predict_file(file, n, temp_loc):
    n_samples = 1000    
    vec_dim   = 400
    
    data = pd.read_table(file,header=None)
    df_char = pd.DataFrame()
    df_word = pd.DataFrame()
    df_par = pd.DataFrame()
    df_stat = pd.DataFrame()    
    res = []
    

    for raw_sample in data[:n].iterrows():
        
        n_values = len(raw_sample)
        if n_samples > n_values:
            n_samples = n_values    

        raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)
        
        df_char = df_char.append(extract_bag_of_characters_features(raw_sample, n_values), ignore_index=True)
        df_word = df_word.append(extract_word_embeddings_features(raw_sample),             ignore_index=True)
        df_par  = df_par.append(infer_paragraph_embeddings_features(raw_sample, vec_dim),  ignore_index=True)
        df_stat = df_stat.append(extract_bag_of_words_features(raw_sample),                ignore_index=True)   

    # temp post-processing:hack to get dimensions match
    stat_col = list(pd.read_csv(os.path.join(SHERLOCKPATH, 'canonical/rest_col.tsv'),sep='\t',header=None)[1])
    char_col = list(pd.read_csv(os.path.join(SHERLOCKPATH, 'canonical/char_col.tsv'), sep='\t', header=None)[1])
    df_large = pd.concat([df_char,df_stat],axis=1)
    df_stat = df_large[stat_col]
    df_char = df_large[char_col]

    #df_char = df_char.iloc[:, 10:-5]
    #df_word['dummy'] = 1.0
    #df_stat = pd.concat([df_stat, df_char.iloc[:, :10], df_char.iloc[:, -5:]], axis=1)  

    df_char.to_csv(os.path.join(temp_loc, 'df_char.csv'), index=False)
    df_word.to_csv(os.path.join(temp_loc, 'df_word.csv'), index=False)
    df_par.to_csv(os.path.join(temp_loc, 'df_par.csv'), index=False)
    df_stat.to_csv(os.path.join(temp_loc, 'df_stat.csv'), index=False)   
    

    ## Load Sherlock model
    # load json and create model
    file = open(os.path.join(SHERLOCKPATH,'NN_model_multiinput_final_val.json'), 'r')
    sherlock_file  = file.read()
    sherlock_model = tensorflow.keras.models.model_from_json(sherlock_file)
    file.close()    

    # load weights into new model
    sherlock_model.load_weights(os.path.join( SHERLOCKPATH, 'NN_weights_multiinput_final_val.h5'))

    # compile model
    sherlock_model.compile(optimizer = 'adam',
                           loss      = 'categorical_crossentropy',
                           metrics   = ['categorical_accuracy'])    

    for i in range(n):  

        label_predicted = sherlock_model.predict([[df_char.values[i,:]], [df_word.values[i,:]], [df_par.values[i,:]], [df_stat.values[i,:]]])
        print(len(label_predicted[0]))
        print(label_predicted, np.argmax(label_predicted))
        res.append(np.argmax(label_predicted))

    return res

