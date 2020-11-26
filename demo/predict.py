import torch
from sklearn.preprocessing import LabelEncoder
import os
from os.path import join
os.environ['LDA_name'] = 'num-directstr_thr-0_tn-400'

import pandas as pd
import numpy as np
from extract.feature_extraction.topic_features_LDA import extract_topic_features
from extract.feature_extraction.sherlock_features import extract_sherlock_features
from utils import get_valid_types
from model import models_sherlock
from model.torchcrf import CRF


TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)
label_enc = LabelEncoder()
label_enc.fit(valid_types)

MAX_COL_COUNT = 6
topic_dim = 400
pre_trained_loc = './pretrained_sato'
device = 'cpu'



feature_group_cols = {}
sherlock_feature_groups = ['char', 'word', 'par', 'rest']
for f_g in sherlock_feature_groups:
    feature_group_cols[f_g] = list(pd.read_csv(join(os.environ['BASEPATH'],
                                          'configs', 'feature_groups', 
                                          "{}_col.tsv".format(f_g)),
                                           sep='\t', header=None, 
                                           index_col=0)[1])



pad_vec = lambda x: np.pad(x, (0, topic_dim - len(x)),
                    'constant',
                    constant_values=(0.0, 1/topic_dim))


# load models
classifier = models_sherlock.build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim, dropout_ratio=0.35)
#classifier.load_state_dict(torch.load(join(pre_trained_loc, 'sherlock_None.pt'), map_location=device))
model = CRF(len(valid_types) , batch_first=True).to(device)
#model.load_state_dict(torch.load(join(pre_trained_loc, 'model.pt'), map_location=device))

loaded_params = torch.load(join(pre_trained_loc, 'model.pt'), map_location=device)
classifier.load_state_dict(loaded_params['col_classifier'])
model.load_state_dict(loaded_params['CRF_model'])

classifier.eval()
model.eval()


def extract(df):

    df_dic = {'df':df, 'locator':'None', 'dataset_id':'None'}
    feature_dic = {}
    n = df.shape[1]

    # topic vectors
    topic_features = extract_topic_features(df_dic)
    topic_vec = pad_vec(topic_features.loc[0,'table_topic'])
    feature_dic['topic'] = torch.FloatTensor(np.vstack((np.tile(topic_vec,(n,1)), np.zeros((MAX_COL_COUNT - n, topic_dim)))))


    # sherlock vectors
    sherlock_features = extract_sherlock_features(df_dic)
    for f_g in feature_group_cols:
        temp = sherlock_features[feature_group_cols[f_g]].to_numpy()
        temp = np.vstack((temp, np.zeros((MAX_COL_COUNT - n, temp.shape[1])))).astype('float')
        temp = np.nan_to_num(temp)
        feature_dic[f_g] = torch.FloatTensor(temp)

    # dictionary of features, labels, masks
    return feature_dic, np.zeros(MAX_COL_COUNT), torch.tensor([1]*n + [0]*(MAX_COL_COUNT-n), dtype=torch.uint8)



def evaluate(df):

    feature_dic, labels, mask = extract(df)

    emissions = classifier(feature_dic).view(1, MAX_COL_COUNT, -1)
    mask = mask.view(1, MAX_COL_COUNT)
    pred = model.decode(emissions, mask)[0]

    return label_enc.inverse_transform(pred)


