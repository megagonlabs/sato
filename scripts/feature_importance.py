from time import time
import os
from os.path import join
import numpy as np
import numpy.ma as ma
import json
import sys
import datetime
import configargparse
from utils import str2bool, str_or_none, name2dic, get_valid_types
import copy
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter
from model.torchcrf import CRF

from model import datasets
from model.models_sherlock import FeatureEncoder, SherlockClassifier, build_sherlock
from sklearn.metrics import classification_report

# =============
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset


def eval_batch_col(classifier, val_dataset, batch_size, device):

    val_batch_generator = datasets.generate_batches_col(val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=True,
                                               device=device)

    y_pred, y_true = [], []
    for batch_idx, batch_dict in enumerate(val_batch_generator):
        y = batch_dict["label"]
        X = batch_dict["data"]

        # Pred
        pred = classifier(X)
        y_pred.extend(pred.cpu().numpy())
        y_true.extend(y.cpu().numpy())

    
    report = classification_report(y_true, np.argmax(y_pred, axis=1), output_dict=True)
    return report

# evaluate and return prediction & true labels of a table batch
def eval_batch(classifier, model, val_dataset, batch_size, device, n_worker, MAX_COL_COUNT):


    validation = datasets.generate_batches(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, 
                                           drop_last=True,
                                           device=device,
                                           n_workers=n_worker)
    y_pred, y_true = [], []
    for table_batch, label_batch, mask_batch in tqdm(validation):
        #pred, labels = eval_batch(table_batch, label_batch, mask_batch)
            
        # reshap (table_batch * table_size * features)
        for f_g in table_batch:
            table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

        emissions = classifier(table_batch).view(batch_size, MAX_COL_COUNT, -1)
        pred = model.decode(emissions, mask_batch)

        pred = np.concatenate(pred)
        labels = label_batch.view(-1).cpu().numpy()
        masks = mask_batch.view(-1).cpu().numpy()
        invert_masks = np.invert(masks==1)
        
        y_pred.extend(pred)
        y_true.extend(ma.array(labels, mask=invert_masks).compressed())

    val_acc = classification_report(y_true, y_pred, output_dict=True)
    return val_acc


importance = lambda x, b: (b['f1-score']-x['f1-score'])/b['f1-score']*100

if __name__ == "__main__":


    #################### 
    # Load configs
    #################### 
    p = configargparse.ArgParser()
    p.add('-c', '--config_file', required=False, is_config_file=True, help='config file path')

    # general configs
    p.add('--n_worker', type=int, default=4, help='# of workers for dataloader')
    p.add('--TYPENAME', type=str, help='Name of valid types', env_var='TYPENAME')
    p.add('--MAX_COL_COUNT', type=int, default=6, help='Max number of columns in a table (padding for batches)') 
    p.add('--table_batch_size', type=int, default=100, help='# of tables in a batch')

    # sherlock configs
    p.add('--sherlock_feature_groups', nargs='+', default=['char','rest','par','word'])
    p.add('--topic', type=str_or_none, default=None)

    # exp configs
    p.add('--batch_size', type=int, default=256, help='# of col in a batch')

    p.add('--corpus_list', nargs='+', default=['webtables1-p1', 'webtables2-p1'])
    p.add('--multi_col_only', type=str2bool, default=False, help='filtering only the tables with multiple columns')
    #p.add('--mode', type=str, help='experiment mode', choices=['train', 'eval'], default='train')
    p.add('--model_path', type=str, help='Load pretrained parameters')
    p.add('--model_type', type=str, choices=['single', 'CRF'], help='for sherlock or CRF models')

    #p.add('--comment', type=str, default='')

    args = p.parse_args()
    print("----------")
    print(args)
    print("----------")
    print(p.format_values())    # useful for logging where different settings came from
    print("----------")


    n_worker = args.n_worker
    TYPENAME = args.TYPENAME

    sherlock_feature_groups = args.sherlock_feature_groups
    topic_name = args.topic

    batch_size = args.batch_size
    corpus_list = args.corpus_list
    
    MAX_COL_COUNT = args.MAX_COL_COUNT if args.model_type=='CRF' else None


    seed_list = [1001, 1002, 1003, 1004, 1005]
    #################### 
    # Preparations
    #################### 
    valid_types = get_valid_types(TYPENAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PyTorch device={}".format(device))

    if topic_name:
        topic_dim = int(name2dic(topic_name)['tn'])
    else:
        topic_dim = None

    if args.topic is not None:
        feature_group_list = args.sherlock_feature_groups + ['topic']
    else:
        feature_group_list = args.sherlock_feature_groups
        

    # 1. Dataset
    t1 = time()
    print("Creating Dataset object...")
    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    # load data through table instance 
    multi_tag = '_multi-col' if args.multi_col_only else ''

    train_test_path = join(os.environ['BASEPATH'], 'extract', 'out', 'train_test_split')
    org_tests, shuffle_tests = [], []

    for corpus in corpus_list:
        with open(join(train_test_path, '{}_{}{}.json'.format(corpus, TYPENAME, multi_tag)), 'r') as f:
            split = json.load(f)
        
        whole_corpus = datasets.TableFeatures(corpus,
                                                sherlock_feature_groups, 
                                                topic_feature=topic_name, 
                                                label_enc=label_enc, 
                                                id_filter=None,
                                                max_col_count=MAX_COL_COUNT)


        shuffle_corpus = datasets.ShuffleFeatures(corpus,
                                                sherlock_feature_groups, 
                                                topic_feature=topic_name, 
                                                label_enc=label_enc, 
                                                id_filter=None,
                                                max_col_count=MAX_COL_COUNT,
                                                shuffle_group=None)

        test = copy.copy(whole_corpus).set_filter(split['test'])
        shuffle_test = copy.copy(shuffle_corpus).set_filter(split['test'])
        if args.model_type == 'single':
            test = test.to_col()
            shuffle_test = shuffle_test.to_col()
        org_tests.append(test)
        shuffle_tests.append(shuffle_test)



    val_dataset = ConcatDataset(org_tests)

    t2 = time()
    print("Done ({} sec.)".format(int(t2 - t1)))

    # create models
    classifier = build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim).to(device)
    model = CRF(len(valid_types) , batch_first=True).to(device)


    if args.model_type == 'single':

        # load pre-trained model
        model_loc = join(os.environ['BASEPATH'],'model','pre_trained_sherlock', TYPENAME)
        classifier.load_state_dict(torch.load(join(model_loc, args.model_path), map_location=device))
        classifier.eval()


        # eval
        with torch.no_grad():
            result_list = []
            # get base accuracy
            report_b = eval_batch_col(classifier, val_dataset, batch_size, device)
            

            for f_g in feature_group_list:

                for corpus in shuffle_tests:
                    corpus = corpus.set_shuffle_group(f_g)

                for seed in seed_list:
                    for corpus in shuffle_tests:
                        corpus.reset_shuffle_seed(seed)

                    report = eval_batch_col(classifier, ConcatDataset(shuffle_tests), batch_size, device)
                    result_list.append([f_g, 'macro avg', importance(report['macro avg'], report_b['macro avg']), seed])
                    result_list.append([f_g, 'weighted avg',importance(report['weighted avg'], report_b['weighted avg']), seed])

            df = pd.DataFrame(result_list, columns=['Feature_group', 'Metric', 'Score', 'Seed'])

            df.to_csv('feature_importance_single_{}.csv'.format(topic_name), index=False)
            print(df)

    elif args.model_type == 'CRF':
        
        # load pre-trained model
        model_loc = join(os.environ['BASEPATH'],'model','pre_trained_CRF', TYPENAME)
        loaded_params = torch.load(join(model_loc, args.model_path), map_location=device)
        classifier.load_state_dict(loaded_params['col_classifier'])
        model.load_state_dict(loaded_params['CRF_model'])

        classifier.eval()
        model.eval()

        # eval
        with torch.no_grad():
            result_list = []
            # get base accuracy
            report_b = eval_batch(classifier, model, val_dataset, args.table_batch_size, device, n_worker, MAX_COL_COUNT)
            

            for f_g in feature_group_list:

                for corpus in shuffle_tests:
                    corpus = corpus.set_shuffle_group(f_g)

                for seed in seed_list:
                    for corpus in shuffle_tests:
                        corpus.reset_shuffle_seed(seed)

                    report = eval_batch(classifier, model, ConcatDataset(shuffle_tests), args.table_batch_size, device, n_worker, MAX_COL_COUNT)
                    result_list.append([f_g, 'macro avg', importance(report['macro avg'], report_b['macro avg']), seed])
                    result_list.append([f_g, 'weighted avg',importance(report['weighted avg'], report_b['weighted avg']), seed])

            df = pd.DataFrame(result_list, columns=['Feature_group', 'Metric', 'Score', 'Seed'])
            df.to_csv('feature_importance_CRF_{}.csv'.format(topic_name), index=False)
            print(df)






