import pandas as pd
import os
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import json
import configargparse

TYPENAME = os.environ['TYPENAME']
header_path = join(os.environ['BASEPATH'], 'extract/out/headers', TYPENAME)


tmp_path = 'out/train_test_split'
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)


p = configargparse.ArgParser()
p.add('-c', '--config_file', is_config_file=True, help='config file path')
p.add('--multi_col_only', type=bool, default=False, help='filtering only the tables with multiple columns')
p.add('--corpus_list', nargs='+', default=['manyeyes', 'opendata', 'plotly', 'webtables1-p1', 'webtables2-p1'])

#corpus_list=['manyeyes', 'opendata', 'plotly', 'webtables1-p1', 'webtables2-p1']
args = p.parse_args()
print("----------")
print(p.format_values())    # useful for logging where different settings came from
print("----------")

corpus_list = args.corpus_list
multi_col = args.multi_col_only

print(corpus_list)

sample_percentages = [20, 40, 60, 80] # Percentage of sampling the training data

for corpus in corpus_list:
    print('Spliting {}'.format(corpus))
    multi_tag = '_multi-col' if multi_col else ''
    
    split_dic = {}

    df = pd.read_csv(join(header_path, "{}_{}_header_valid.csv".format(corpus, TYPENAME)))
    if multi_col:
        df.loc[:, 'col_count'] = df.apply(lambda x: len(eval(x['field_names'])), axis=1)
        df= df[df['col_count']>1]

    df.loc[:,'table_id'] = df.apply(lambda x: '+'.join([x['locator'], x['dataset_id']]), axis = 1)
    train_list, test_list= train_test_split(df['table_id'], test_size=0.2, random_state=42)
    split_dic = {'train':list(train_list), 'test':list(test_list)}

    total_size = len(split_dic['train'])
    sample_from = split_dic['train']
    for s_p in sorted(sample_percentages, reverse=True):

        sample_size = int(s_p/100*total_size)
        
        new_sample = np.random.choice(sample_from, sample_size, replace=False)
        sample_from = new_sample
        split_dic['train_{}per'.format(s_p)] = list(new_sample)

    print(split_dic.keys())


    print("Done, {} training tables{}, {} testing tables{}".format(
                                                                len(split_dic['train']),
                                                                multi_tag,
                                                                len(split_dic['test']),
                                                                multi_tag))
    with open(join(tmp_path, '{}_{}{}.json'.format(corpus, TYPENAME, multi_tag)),'w') as f:
        json.dump(split_dic, f)