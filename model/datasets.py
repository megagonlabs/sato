import json
import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from os.path import join
from utils import  name2dic, get_valid_types, load_tmp_df

import torch
from torch.utils.data import Dataset, DataLoader


# global dataset settings
EMBEDDING_FILL= 0.0
SEED = 100


TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)

header_path = join(os.environ['BASEPATH'], 'extract', 'out', 'headers', TYPENAME)
feature_path = join(os.environ['BASEPATH'], 'extract', 'out', 'features', TYPENAME)

# load column headers
feature_group_cols = {}
sherlock_feature_groups = ['char', 'word', 'par', 'rest']
other_feature_groups = ['topic']
for f_g in sherlock_feature_groups + other_feature_groups:
    feature_group_cols[f_g] = list(pd.read_csv(join(os.environ['BASEPATH'],
                                          'configs', 'feature_groups', 
                                          "{}_col.tsv".format(f_g)),
                                           sep='\t', header=None, 
                                           index_col=0)[1])

def generate_batches_col(dataset,
                     batch_size,
                     shuffle=True,
                     drop_last=True,
                     device="cpu"):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)

    for data_dict in dataloader:
        if device == "cpu":
            yield data_dict
        else:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                if type(tensor) == dict:
                    out_data_dict[name] = {}
                    for n, t in tensor.items():
                        out_data_dict[name][n] = data_dict[name][n].to(device)
                else:
                    out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

#generate table batches
def generate_batches(dataset,
                     batch_size,
                     shuffle=True,
                     drop_last=True,
                     device="cpu",
                     n_workers=0):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=n_workers,
                            pin_memory=False)

    for data_dict, labels, masks in dataloader:

        if device == "cpu" or device == torch.device('cpu'):
            yield data_dict, labels, masks
        else:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device, non_blocking=True)
            yield out_data_dict, labels.to(device, non_blocking=True), masks.to(device, non_blocking=True)


tmp_path = join(os.environ['BASEPATH'], 'tmp')
if not os.path.exists(tmp_path):
    print("tmp directory not found. create.")
    os.makedirs(tmp_path)


class TableFeatures(Dataset):
    '''
    Feature vectors organized in terms of tables.
    For a table with n columns, returns: 
    features_dic: dictionary of n x M tensors 
    labels: n x 1 tensor with encoded labels
    max_col_count need to be set for batch_size >1
    '''
    def __init__(self,
                 corpus,
                 sherlock_features: List[str] = None,
                 topic_feature: str = None,
                 label_enc: LabelEncoder = None,
                 id_filter: List[str] = None,
                 max_col_count:int = None): # if not None, pad the returning tensors to max_col_count columns. 
        
        self.sherlock_features = sherlock_features # list of sherlock features
        self.topic_feature = topic_feature # name of topic_feature
        self.label_enc = label_enc
        self.max_col_count = max_col_count
        
        self.df_header = load_tmp_df(header_path, tmp_path, '{}_{}_header_valid'.format(corpus,TYPENAME), table=True)

        # filter training/testing sets
        # filtering won't affect the pickled file used or the dictionary loaded
        if id_filter is not None:
            self.df_header = self.df_header.loc[id_filter]

        self.data_dic = {}
        
        start = time.time()

        sherlock_loaded = False

        if len(sherlock_features) > 0:

            for f_g in sherlock_features:

                dic_pkl_file = join(tmp_path, '{}_{}_{}.pkl'.format(corpus, TYPENAME, f_g))
                if os.path.exists(dic_pkl_file):
                    with open(dic_pkl_file, "rb") as fin:
                        self.data_dic[f_g] = pickle.load(fin)
                else:
                    if not sherlock_loaded:
                        self.df_sherlock = load_tmp_df(feature_path, tmp_path, '{}_{}_sherlock_features'.format(corpus, TYPENAME), table=False)
                        sherlock_loaded = True

                    print("Prepare seperate files for feature groups...")
                    feat_dict = self.df_sherlock.groupby('table_id').apply(lambda x: x[feature_group_cols[f_g]].fillna(EMBEDDING_FILL).values.astype('float')).to_dict()
                    for i in feat_dict:
                        feat_dict[i] = torch.FloatTensor(feat_dict[i])
                    self.data_dic[f_g] = feat_dict

                    with open(dic_pkl_file, "wb") as fout:
                        pickle.dump(feat_dict, fout, protocol=2)


        if topic_feature is not None:
            self.topic_no = int(name2dic(self.topic_feature)['tn'])

            dic_pkl_file = join(tmp_path, '{}_{}_{}.pkl'.format(corpus, TYPENAME, topic_feature))
            if os.path.exists(dic_pkl_file):
                with open(dic_pkl_file, "rb") as fin:
                    self.data_dic['topic'] = pickle.load(fin)

            else:

                fill = 1.0/self.topic_no
                pad_vec = lambda x: np.pad(x, (0, self.topic_no - len(x)),
                                    'constant',
                                    constant_values=(0.0, fill))

                self.df_topic = load_tmp_df(feature_path, tmp_path, '{}_{}_topic-{}_features'.format(corpus, TYPENAME, topic_feature), table=True)
                topic_dict = self.df_topic.apply(lambda x: pad_vec(eval(x['table_topic'])), axis=1).to_dict()

                for i in topic_dict:
                    topic_dict[i] = torch.FloatTensor(topic_dict[i])

                self.data_dic['topic'] = topic_dict
                with open(dic_pkl_file, "wb") as fout:
                    pickle.dump(topic_dict, fout, protocol=2)


        
        print("Total data preparation time:", time.time()-start)


    def __len__(self):
        return len(self.df_header)
    
    def __getitem__(self, idx):
        
        features_dic = {}

        table_id = self.df_header.index[idx]
        labels = [valid_types[i] for i in eval(self.df_header.loc[table_id]['field_names'])]

        # pad the tensor for batches and create mask
        if self.max_col_count is not None:
            col_count = len(labels)
            mask = np.zeros(self.max_col_count, dtype=int)
            mask[:col_count].fill(1)
            mask = torch.tensor(mask, dtype=torch.uint8)
            pad = (0,0,0, self.max_col_count - col_count)
            new_col_count = self.max_col_count
        else:
            mask = torch.zeros(len(labels)) # need to be a tensor for batch generation
            pad = None
            new_col_count = len(labels)

            
        if len(self.sherlock_features) > 0:
            
            for f_g in self.sherlock_features:
                try:
                    if pad is not None:
                        features_dic[f_g] = F.pad(self.data_dic[f_g][table_id], pad, 'constant', 0.0)
                    else:
                        features_dic[f_g] = self.data_dic[f_g][table_id]

                except Exception as e:
                    print("Exception sherlock feature", e)
                    # fill in default feature vector
                    features_dic[f_g] = torch.full((new_col_count, len(feature_group_cols[f_g])), EMBEDDING_FILL, dtype=torch.float)


        if self.topic_feature:
            try:
                features_dic['topic']  = self.data_dic['topic'][table_id].repeat(new_col_count, 1)

            except Exception as e:
                print("Exception topic feature", e)
                features_dic['topic'] = torch.full((new_col_count, self.topic_no), 1.0/self.topic_no, dtype=torch.float)


        return features_dic, np.pad(self.label_enc.transform(labels), (0, new_col_count - len(labels)), 'constant', constant_values=(-1,-1)) , mask

    def set_filter(self, id_filter):
        self.df_header = self.df_header.loc[id_filter]
        return self


    def to_col(self):
        # create column feature instance (SherlockDataset)
        start = time.time()
        col_dic = {}
        table_ids = list(self.df_header.index)
        labels = np.concatenate([eval(x) for x in list(self.df_header.field_names)])
        col_counts = {table:len(eval(self.df_header.loc[table].field_names)) for table in table_ids}
        for f_g in self.data_dic:
            feature_dic = self.data_dic[f_g]
            if f_g=='topic':

                col_dic[f_g] = torch.cat([feature_dic[table].repeat(col_counts[table],1) for table in table_ids])
            else:  
                col_dic[f_g] = torch.cat([feature_dic[table] for table in table_ids])

        print("Time used to convert to SherlockDataset (column features)", time.time()-start)
        return SherlockDataset(tensor_dict=col_dic, labels=[valid_types[i] for i in labels], label_enc=self.label_enc)



class SherlockDataset(Dataset):
    def __init__(self,
                 df_dict: Dict[str, pd.DataFrame]=None,
                 tensor_dict: Dict[str, torch.FloatTensor]=None,
                 labels: List[str] =[],
                 label_enc: LabelEncoder = None):  
        assert not (df_dict is None and tensor_dict is None),\
            print('df_dict and tensor_dict can\'t be both None')

        assert len(labels)>0, 'lables can\'t be empty'

        if label_enc is None:
            label_enc = LabelEncoder()
            label_enc.fit(labels)
        self.label_enc = label_enc
        self.label_ids = self.label_enc.transform(labels)

        if tensor_dict is not None:
            self.name_tensor_dict = tensor_dict
            self.f_g_names = list(tensor_dict.keys())
            self.len = tensor_dict[self.f_g_names[0]].shape[0]
        else: 

            self.f_g_names = df_dict.keys()
            self.len = len(list(df_dict.values())[0])

            # df_dict must have at least one key-value pair
            assert len(df_dict) > 0
            # Make sure each df has the same size
            for name, df in df_dict.items():
                assert len(df) == len(list(df_dict.values())[0])

            # Convert dataframe into a dictionary of FloatTensor to avoid on-the-fly conversion
            self.name_tensor_dict = {}
            for name, df in df_dict.items():
                self.name_tensor_dict[name] = torch.FloatTensor(df.values.astype('float'))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {"label": self.label_ids[idx],
                "data": dict([
                    [name,
                     self.name_tensor_dict[name][idx]] for name in self.f_g_names])}




class ShuffleFeatures(TableFeatures):
    # Special dataset class for feature importance analysis
    # shuffle features in certain feature group 

    def __init__(self,
                 corpus,
                 sherlock_features: List[str] = None,
                 topic_feature: str = None,
                 label_enc: LabelEncoder = None,
                 id_filter: List[str] = None,
                 max_col_count:int = None,
                 shuffle_group:str=None):

        super().__init__(corpus,
                                    sherlock_features,
                                    topic_feature,
                                    label_enc,
                                    id_filter,
                                    max_col_count)

        l = len(self.df_header)
        self.tempcorpus = corpus

        self.shuffle_group = shuffle_group
        self.prng = np.random.RandomState(SEED)
        self.shuffle_order = self.prng.permutation(l)
 

    def __getitem__(self, idx):
        assert self.shuffle_group is not None, 'Feature group to shffule cannot be None'
        #print(self.tempcorpus, idx, self.shuffle_order[idx])
        features_dic, labels, mask = super().__getitem__(idx)
        # shuffle f_g, the # of columns in tables could be different
        new_feature_dic, _, _ = super().__getitem__(self.shuffle_order[idx])
        features_dic[self.shuffle_group] = new_feature_dic[self.shuffle_group]

        return features_dic, labels, mask

    def set_shuffle_group(self, group):
        self.shuffle_group = group

    def reset_shuffle_seed(self, seed):
        prng = np.random.RandomState(seed)
        self.shuffle_order = prng.permutation(self.__len__())

    def set_filter(self, id_filter):
        self.df_header = self.df_header.loc[id_filter]
        # reset shuffling order after filtering
        self.shuffle_order = self.prng.permutation(len(self.df_header))
        return self

    def to_col(self):
        # create column feature instance (SherlockDataset)
        start = time.time()
        col_dic = {}
        table_ids = list(self.df_header.index)
        labels = np.concatenate([eval(x) for x in list(self.df_header.field_names)])
        col_counts = {table:len(eval(self.df_header.loc[table].field_names)) for table in table_ids}
        for f_g in self.data_dic:
            feature_dic = self.data_dic[f_g]
            if f_g=='topic':

                col_dic[f_g] = torch.cat([feature_dic[table].repeat(col_counts[table],1) for table in table_ids])
            else:  
                col_dic[f_g] = torch.cat([feature_dic[table] for table in table_ids])

        print("Time used to convert to ShuffleFeaturesCol (column features)", time.time()-start)
        return ShuffleFeaturesCol(tensor_dict=col_dic, labels=[valid_types[i] for i in labels], label_enc=self.label_enc)
        

class ShuffleFeaturesCol(SherlockDataset):
    # Special dataset class for feature importance analysi
    # shuffle features in certain feature group 

    def __init__(self,
                 df_dict: Dict[str, pd.DataFrame]=None,
                 tensor_dict: Dict[str, torch.FloatTensor]=None,
                 labels: List[str] =[],
                 label_enc: LabelEncoder = None,
                 shuffle_group: str = None):

        super().__init__(df_dict,
                         tensor_dict,
                         labels,
                         label_enc)

        l = self.__len__()

        self.shuffle_group = shuffle_group
        prng = np.random.RandomState(SEED)
        self.shuffle_order = prng.permutation(l)
 

    def __getitem__(self, idx):
        assert self.shuffle_group is not None, 'Feature group to shffule cannot be None'

        dic = {}
        for name in self.f_g_names:
            if name == self.shuffle_group:
                dic[name] = self.name_tensor_dict[name][self.shuffle_order[idx]]
            else:
                dic[name] = self.name_tensor_dict[name][idx]


        return {"label": self.label_ids[idx],
                "data": dic}

    def set_shuffle_group(self, group):
        self.shuffle_group = group

    def reset_shuffle_seed(self, seed):
        prng = np.random.RandomState(seed)
        self.shuffle_order = prng.permutation(self.__len__())


if __name__ == '__main__':
    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    topic = 'thr-0_num-directstr_features'
    t = TableFeatures('webtables1-p1', ['char','rest','word', 'par'], topic_feature=None, label_enc=label_enc)
    tl = generate_batches(t, 1, True)

    for i in range(3):
        print(next(tl))
