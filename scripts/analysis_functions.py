from os.path import join
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import os

#import seaborn as sns



def get_predictions(path):
    # helper to retrieve the predictions
    last_epoch = max([int((x.split('.')[0]).split('_')[3]) for x in os.listdir(path) if x.startswith('y_pred_epoch')])
    y_sherlock = np.load(join(path, 'y_pred_sherlock.npy'))
    y_pred = np.load(join(path, 'y_pred_epoch_{}.npy'.format(last_epoch)))
    y_true = np.load(join(path, 'y_true.npy'))
    
    return y_sherlock, y_pred, y_true


def report_gen(y_pred, y_true, report_name=None, out_loc=join(os.environ['BASEPATH'], 'results', 'report')):
    # given predicted and true labels, 
    # generate the overall results and pertype analysis with misclassification
    report = classification_report(y_true, y_pred, output_dict=True)

    df_report = pd.DataFrame(columns=['type', 'precision', 'recall','f1-score', 'support'])

    overall = {}
    for t in report:
        if t not in ['accuracy','macro avg','weighted avg']:
            report[t]['type']=t
            df_report= df_report.append(report[t], ignore_index=True)
        else:
            overall[t] = report[t]


    # extract misclassification details
    dic = {}
    for t,p in zip(y_true, y_pred):
        if t not in dic:
            dic[t] = {'mis_to':{}, 'mis_from':{}}
        if p not in dic:
            dic[p] = {'mis_to':{}, 'mis_from':{}}
            
        if t!=p:
            dic[t]['mis_to'][p] = dic[t]['mis_to'].get(p, 0) + 1
            dic[p]['mis_from'][t] = dic[p]['mis_from'].get(t, 0) + 1

    def first_five(dic):
        return sorted(dic.items(), key=lambda x: x[1], reverse=True)[:5]

    df_report['mis_from_top5'] = df_report.apply(lambda x: first_five(dic[x['type']]['mis_from']),axis=1) # precision
    df_report['mis_to_top5'] = df_report.apply(lambda x: first_five(dic[x['type']]['mis_to']),axis=1) # recall

    # save results
    if report_name is not None:
        if not os.path.exists(out_loc):
            os.mkdir(out_loc)

        df_report.sort_values(['f1-score'], ascending=False).to_csv(join(out_loc,'results_per_type_{}.csv'.format(report_name)))

        with open(join(out_loc, 'overall_{}.json'.format(report_name)), 'w') as outfile:  
            json.dump(overall, outfile)

    return overall, df_report




def per_type_plot(result_A, result_B, name_A, name_B, comment=None,
                  path=join(os.environ['BASEPATH'], 'results', 'figs')):
    # produce plot that compare the per-type f1 of two apporaches.
    
    def melt_df(df):
        df = pd.melt(df, id_vars=['type'], value_vars=['f1-score_{}'.format(name_A),'f1-score_{}'.format(name_B)])
        return df 
    colors = ["windows blue","amber","faded green","dusty purple",  "greyish"]
    color_palette = sns.xkcd_palette(colors)

    overall_A, df_A = result_A
    overall_B, df_B = result_B
    
    df = pd.merge(df_A, df_B, on=['type'], suffixes=("_"+name_A, "_"+name_B))

    better = df[df['f1-score_{}'.format(name_A)] > df['f1-score_{}'.format(name_B)]] # A better
    worse = df[df['f1-score_{}'.format(name_A)] < df['f1-score_{}'.format(name_B)]] #worse
    
    better = better.sort_values(['f1-score_{}'.format(name_B)], ascending=False)
    worse = worse.sort_values(['f1-score_{}'.format(name_B)], ascending=False)
    
    f, axes = plt.subplots(1, 3, figsize=(15, 10))
    f.suptitle( 'Weighted F1-score:      {0} {2:6.3f},  {1} {3:6.3f}, + {6:3.1f}%\n\
                 Marco Average F1-score: {0} {4:6.3f},  {1} {5:6.3f}, + {7:3.1f}%\
                '.format(name_A, name_B,
                         overall_A['weighted avg']['f1-score'],
                         overall_B['weighted avg']['f1-score'],
                         overall_A['macro avg']['f1-score'],
                         overall_B['macro avg']['f1-score'],
                         100*(overall_A['weighted avg']['f1-score'] - overall_B['weighted avg']['f1-score']),
                         100*(overall_A['macro avg']['f1-score'] - overall_B['macro avg']['f1-score']),
                        ), fontsize=16)
    
    sns.barplot(x="value",
                y="type",
                hue ='variable',
                data=melt_df(better),
                ax=axes[0],
                palette=color_palette).set_title("{} better".format(name_A))

    sns.barplot(x="value",
                y="type",
                hue ='variable',
                data=melt_df(worse),
                ax=axes[1],
                palette=color_palette).set_title("{} better".format(name_B))
    

    postfix = '' if comment is None else "_"+comment

    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(join(path, "{}_VS_{}{}.png".format(name_A, name_B, postfix)))
    return 


#####################################################
# Generating tables comparison 
# sherlock vs. +LDA, + CRF and + both
#####################################################



def CRF_tuple_gen(path, LDA_tag):
    
    tuple_list = []

    y_sherlock, y_pred, y_true = get_predictions(path)

#    last_epoch = max([int((x.split('.')[0]).split('_')[3]) for x in os.listdir(path) if x.startswith('y_pred_epoch')])
#    y_sherlock = np.load(join(path, 'y_pred_sherlock.npy'))
#    y_pred = np.load(join(path, 'y_pred_epoch_{}.npy'.format(last_epoch)))
#    y_true = np.load(join(path, 'y_true.npy'))

    res_sherlock = report_gen(y_sherlock, y_true)[0]
    res_CRF = report_gen(y_pred, y_true)[0]
    tuple_list.append((LDA_tag, 'sherlock', res_sherlock['macro avg']['f1-score'], res_sherlock['weighted avg']['f1-score']))
    tuple_list.append((LDA_tag, '+CRF', res_CRF['macro avg']['f1-score'], res_CRF['weighted avg']['f1-score']))
    return tuple_list

def large_table_gen(path, path_LDA):
    tuple_list = CRF_tuple_gen(path,'sherlock') + CRF_tuple_gen(path_LDA, '+LDA')
    df = pd.DataFrame(tuple_list, columns=['LDA', 'CRF', 'macro avg', 'weighted avg'])

    table = df.pivot(index= 'CRF', columns='LDA', values =['macro avg', 'weighted avg'])
    # reindex the columns(multi-index) and rows
    table = table.reindex(['sherlock', '+LDA'], level=1, axis=1).reindex(['sherlock', '+CRF'])
    return table



# generate all per-type plots: sherlock vs. +LDA/ +CRF/ +CRF&LDA
def data_gen_all(path, path_LDA, comment, loc):
    if not os.path.exists(loc):
        os.makedirs(loc)

    y_sherlock, y_CRF, y_true = get_predictions(path)
    y_LDA, y_CRF_LDA, y_true = get_predictions(path_LDA)
    result_sherlock = report_gen(y_sherlock, y_true)
    result_LDA = report_gen(y_LDA, y_true)
    result_CRF = report_gen(y_CRF, y_true)
    result_CRF_LDA = report_gen(y_CRF_LDA, y_true)
    #print( result_sherlock[1])

    result_sherlock[1].to_csv(join(loc, 'result_sherlock_{}.csv'.format(comment)))
    result_LDA[1].to_csv(join(loc, 'result_LDA_{}.csv'.format(comment)))
    result_CRF[1].to_csv(join(loc, 'result_CRF_{}.csv'.format(comment)))
    result_CRF_LDA[1].to_csv(join(loc, 'result_CRF_LDA_{}.csv'.format(comment)))
    
    #per_type_plot(result_LDA, result_sherlock , 'sherlock+LDA', 'Sherlock', comment=comment)    
    #per_type_plot(result_CRF, result_sherlock , 'sherlock+CRF', 'Sherlock', comment=comment)    
    #per_type_plot(result_CRF_LDA, result_sherlock , 'sherlock+CRF&LDA', 'Sherlock', comment=comment)  

