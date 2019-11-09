import pandas as pd
from collections import OrderedDict
import numpy as np
import random
from extract.helpers import utils
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words      import extract_bag_of_words_features
from sherlock.features.word_embeddings   import extract_word_embeddings_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features



n_samples = 1000
vec_dim   = 400

def extract_sherlock_features(df_dic):
    df, locator, dataset_id = df_dic['df'], df_dic['locator'], df_dic['dataset_id']

    all_field_features = []
    
    for i in range(len(df.columns)):
        single_field_feature_set = OrderedDict()

        all_field_features.append(single_field_feature_set)


    for field_order, field_name in enumerate(df.columns):
        v = df[field_name]

        field_id = field_order
        all_field_features[field_order]['locator'] = locator
        all_field_features[field_order]['dataset_id'] = dataset_id
        all_field_features[field_order]['field_id'] = '{}:{}'.format(dataset_id, field_id) # field id in the filterd table
        all_field_features[field_order]['header'] = field_name
        all_field_features[field_order]['header_c'] = utils.canonical_header(field_name)
        n_values = len(v)

        try:
            try:
                field_values = list(v[:v.last_valid_index()])
            except Exception as e:
                field_values = v
                continue
            # sample if more than 1000 values in column
            if n_values > n_samples:
                n_values = n_samples
                v = random.choices(v, k=n_values)
            raw_sample = pd.Series(v).astype(str)

            f_ch = extract_bag_of_characters_features(raw_sample, n_values)
            f_word = extract_word_embeddings_features(raw_sample)
            f_par = infer_paragraph_embeddings_features(raw_sample, vec_dim)
            f_stat = extract_bag_of_words_features(raw_sample)
            for feature_set in [ f_ch, f_word, f_par, f_stat ]:
                for k, v in feature_set.items():
                    all_field_features[field_order][k] = v


        except Exception as e:
            print('Single field exception:', e)
            continue


    return pd.DataFrame(all_field_features)
