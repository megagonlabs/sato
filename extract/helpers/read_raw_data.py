'''
Helper functions to read through raw data for each corpus. 
Each is a generator that yields a single dataset and metadata in the form:

{
    'df'
    'locator',
    'dataset_id'
}
'''
import os
from os import listdir
from os.path import join
from collections import OrderedDict
import argparse
import gzip
import json
import chardet
import traceback
import itertools
import numpy as np
import pandas as pd
from .general_helpers import clean_chunk
from .type_detection import detect_field_type, data_type_to_general_type, data_types, general_types

raw_data_dir = os.environ['RAW_DIR']
CHUNK_SIZE = 500

all_corpus = ['plotly','manyeyes','test','opendata'] + ['webtables{}'.format(i) for i in range(10)]

data_dirs = { x: join(raw_data_dir, 'webtables', x) if x.startswith('webtables') else join(raw_data_dir, x)  for x in all_corpus}

def listdir_not_hidden(path):
    dirlist = listdir(path)
    return filter(lambda x: not x.startswith('.'), dirlist) if dirlist else []


def plotly_file_iter(file_path):
    # large file, read in chunks
    raw_df_chunks = pd.read_csv(
            file_path,
            sep='\t',
            usecols=['fid', 'table_data', 'layout', 'chart_data'],
            error_bad_lines=False,
            warn_bad_lines=False,
            chunksize=CHUNK_SIZE,
            encoding='utf-8'
        )

    for chunk_num, chunk in enumerate(raw_df_chunks):
        chunk = clean_chunk(chunk)  
        for row in chunk.iterrows():
            yield row

def extract_plotly(table_data, locator, dataset_id, exact_num_fields=None, min_fields=None, max_fields=None, valid_fields=None):

    #table_data = chart_obj.table_data
    fields = table_data[list(table_data.keys())[0]]['cols']  
    sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
    num_fields = len(sorted_fields)

    if exact_num_fields:
        if num_fields != exact_num_fields: return
    if min_fields:
        if num_fields < min_fields: return                   
    if max_fields:
        if num_fields > max_fields: return

    data_as_dict = OrderedDict()
    for k, v in sorted_fields:
        data_as_dict[k] = pd.Series(v['data'])

    df = pd.DataFrame(data_as_dict)

    # If specified, only return the valid fields
    if valid_fields is not None:
        df = df.iloc[:,valid_fields]

    result = {
        'df': df,
        'dataset_id': dataset_id,
        'locator': locator
    }
    return result

def get_plotly_dfs(limit=None, exact_num_fields=None, min_fields=None, max_fields=None):
    corpus = 'plotly'
    base_dir = data_dirs[corpus]
    files = [ f for f in listdir_not_hidden(base_dir) if f.endswith('.tsv') ]
    for f in files[:limit]:
        file_path = join(data_dirs[corpus], f)
        plotly_row_iter = plotly_file_iter(file_path)

         
        for chart_num, chart_obj in plotly_row_iter:

            df = extract_plotly(chart_obj.table_data, f, chart_obj.fid, exact_num_fields, min_fields, max_fields)
            if df is not None:
                yield df


def load_manyeyes(full_file_path, locator, dataset_id, exact_num_fields=None, min_fields=None, max_fields=None, valid_fields=None):
    try:
        df = pd.read_csv(
            full_file_path,
            error_bad_lines=False,
            warn_bad_lines=False,
            sep='\t',
            encoding='utf-8'
        )

        num_fields = len(df.columns)

        # If specified, only return the valid fields
        if valid_fields is not None:
            df = df.iloc[:,valid_fields]

        if exact_num_fields:
            if num_fields != exact_num_fields: return
        if min_fields:
            if num_fields < min_fields: return
        if max_fields:
            if num_fields > max_fields: return

        result = {
            'df': df,
            'dataset_id': dataset_id,
            'locator': locator
        }
        return result

    except Exception as e:
        #print("Exception loading manyeyes data", e)
        return

def get_manyeyes_dfs(exact_num_fields=None, min_fields=None, max_fields=None):
    corpus='manyeyes'
    base_dir = data_dirs[corpus]
    files = []
    for year_dir in listdir_not_hidden(base_dir):
        for month_dir in listdir_not_hidden(join(base_dir, year_dir)):
            month_files = listdir_not_hidden(join(base_dir, year_dir, month_dir))
            files.append([ year_dir, month_dir, month_files ])

    for (year_dir, month_dir, month_files) in files:
        for i, file_name in enumerate(month_files):
            locator = join(year_dir, month_dir, file_name)

            full_file_path = join(base_dir, year_dir, month_dir, file_name)
            dataset_id = file_name

            df = load_manyeyes(full_file_path, locator, dataset_id, exact_num_fields, min_fields, max_fields)
            if df:
                yield df 
            else:
                continue



def webtables_iter(path):
    # generate the next line of json(table)
    with gzip.open(path, 'rb') as f_in:
        iter_count = 0  # only count the # of succesfully yield dataframes
        for line_count, dataset in enumerate(f_in):
            try:
                data = json.loads(dataset.decode('utf-8'))
                yield (iter_count, data)
                iter_count+=1
            except UnicodeDecodeError:
                encoding = chardet.detect(dataset)['encoding']
                try:
                    data = json.loads(dataset.decode(encoding))
                    yield (iter_count, data)
                    iter_count+=1
                except Exception as e:
                    #print('Cannot parse:', e)
                    continue
                continue

def extract_webtables(data, locator, dataset_id=None, exact_num_fields=None, min_fields=None, max_fields=None, valid_fields=None, line_no=0):
    # if dataset_id is set, only extract if there's a match
    try:
        # webtables are not uniquely identified by pageTitle + tableNum,
        # TO distinguish between tables, add a index with respect to location in the conatining file.
        if data['hasHeader'] and (data['headerPosition'] == 'FIRST_ROW'):
            d_id = '{}-{}-{}'.format(line_no, data['pageTitle'], data['tableNum'])   

            # table name not matching 
            if dataset_id is not None and d_id != dataset_id:
                return 
            header_row_index = data.get('headerRowIndex', 0)
            data_as_dict = OrderedDict()
            for raw_cols in data['relation']:
                header_row = raw_cols[header_row_index]
                raw_cols.pop(header_row_index)

                parsed_values = pd.Series([ None if (v == '-') else v for v in raw_cols ])
                try:
                    parsed_values = pd.to_numeric(parsed_values, errors='raise')
                except:
                    #print('CAN"T PARASE')
                    pass
                #parsed_values = parsed_values.replace(value='-', None)
                data_as_dict[header_row] = parsed_values

            df = pd.DataFrame(data_as_dict)

            num_fields = len(df.columns)

            if exact_num_fields:
                if num_fields != exact_num_fields: return
            if min_fields:
                if num_fields < min_fields: return
            if max_fields:
                if num_fields > max_fields: return
            
            # If specified, only return the valid fields
            if valid_fields is not None:
                df = df.iloc[:,valid_fields]

            result = {
                'df': df,
                'dataset_id': d_id,
                'locator': locator
            }
            return result
        else:
            return 
    except Exception as e:
        print("Exception in table extraction: ",e)
        return   

def get_webtables_dfs(detailed_name, exact_num_fields=None, min_fields=None, max_fields=None):
    corpus = detailed_name
    base_dir = data_dirs[corpus]    
    files = []
    for sub_dir in listdir_not_hidden(base_dir):
        if sub_dir.endswith(tuple(['.gz', '.lst', '.html'])): continue
        json_files = listdir_not_hidden(join(base_dir, sub_dir, 'warc'))
        files.append([ base_dir, sub_dir, json_files ])

    for (base_dir, sub_dir, json_files) in files:
        for i, file_name in enumerate(json_files):
            full_file_path = join(base_dir, sub_dir, 'warc', file_name)


            locator = join(sub_dir, 'warc', file_name)
            w_iter = webtables_iter(full_file_path)
            
            for idx, data in w_iter:

                df = extract_webtables(data, locator, None, exact_num_fields, min_fields, max_fields, line_no = idx)
                if df is not None:
                    yield df




def load_opendata(full_dataset_path, locator, dataset_id, exact_num_fields=None, min_fields=None, max_fields=None, valid_fields=None):
    engine = 'c'
    encoding = 'utf-8'
    sep=','
    attempts = 2
    while attempts > 0: # don't try forever...
        try:
            df = pd.read_csv(
                full_dataset_path,
                engine=engine,  # https://github.com/pandas-dev/pandas/issues/11166
                error_bad_lines=False,
                warn_bad_lines=False,
                encoding=encoding,
                sep=sep
            )
            num_fields = len(df.columns)

            if num_fields == 1 and sep != ':':
                if sep == ',': sep=';'
                elif sep == ';': sep='\t'
                elif sep == '\t': sep=':'
                attempts -=1
            elif num_fields == 1 and sep == ':':
                with open(full_dataset_path, 'r') as f:
                    head = [next(f) for x in range(100)]
                    head = ''.join(head)
                    for t in [ '<body>', 'html', 'DOCTYPE' ]:
                        if t in head:
                            print('is html')
                            return
                return result

            else:
                if exact_num_fields:
                    if num_fields != exact_num_fields: return #continue
                if max_fields:
                    if num_fields > max_fields: return #continue

                # If specified, only return the valid fields
                if valid_fields is not None:
                    df = df.iloc[:,valid_fields]

                result = {
                    'df': df,
                    'dataset_id': dataset_id,
                    'locator': locator
                }
                return result

                #yield result
                #break

        except UnicodeDecodeError as ude:
            #print("Endcoding error:", encoding, ude)
            encoding = 'latin-1'
            attempts -= 1
        except pd.errors.ParserError as cpe:
            #print('Engeine error:', cpe)
            engine = 'python'
            attempts -= 1
        except Exception as e:
            #print('Exception:', e)
            return

def get_opendata_dfs(exact_num_fields=None, min_fields=None, max_fields=None, valid_fields=None):
    corpus = 'opendata'
    base_dir = data_dirs[corpus]       
    files = []
    for portal_dir in listdir_not_hidden(base_dir):
        full_portal_dir = join(base_dir, portal_dir)
        for dataset_id_dir in listdir_not_hidden(full_portal_dir):
            full_dataset_id_dir = join(full_portal_dir, dataset_id_dir)
            for dataset_name in listdir_not_hidden(full_dataset_id_dir):
                full_dataset_path = join(full_dataset_id_dir, dataset_name)
                locator = join(portal_dir, dataset_id_dir)
                dataset_id = dataset_name

                df = load_opendata(full_dataset_path, locator, dataset_id, exact_num_fields, min_fields, max_fields)
                if df:
                    yield df 


##################################################################
# Filtered iterators (based on the header_iter passed in)
##################################################################
def get_opendata_filtered_dfs(header_iter, exact_num_fields=None, min_fields=None, max_fields=None):
    corpus = 'opendata'
    base_dir = data_dirs[corpus]       
    
    for next_line in header_iter:
        if next_line == 'EOF':
            return
        idx, row = next_line
        locator = row['locator']
        dataset_id = row['dataset_id']
        fields = eval(row['field_list']) #convert string to list
        full_dataset_path = join(base_dir, locator, dataset_id)

        df = load_opendata(full_dataset_path, locator, dataset_id, exact_num_fields, min_fields, max_fields, fields)
        if df:
            yield df


def get_webtables_filterd_dfs(corpus, header_iter, exact_num_fields=None, min_fields=None, max_fields=None):
    base_dir = data_dirs[corpus] # webtables0/ webtables1 ...

    row_idx, row = next(header_iter)

    prev_locator = row['locator']
    prev_dataset_id = row['dataset_id']

    prev_idx = int(prev_dataset_id.split('-')[0])

    w_iter = webtables_iter(join(base_dir, prev_locator))


    _, data = next(itertools.islice(w_iter, prev_idx, None))

    df = extract_webtables(data, prev_locator, prev_dataset_id, exact_num_fields, min_fields, max_fields, list(eval(row['field_list'])), line_no=prev_idx)
    if df is not None:
        yield df


    for header_line in header_iter:
        if header_line =='EOF':
            return
        _, row = header_line
        locator, dataset_id = row['locator'], row['dataset_id']
        new_idx = int(dataset_id.split('-')[0])
        #print("new_idx", new_idx)

        if locator == prev_locator:
            offset = new_idx-prev_idx-1
        else:
            w_iter = webtables_iter(join(base_dir, locator))
            offset = new_idx

        #print('offset', offset)
        _, data = next(itertools.islice(w_iter, offset, None))
        df = extract_webtables(data, locator, dataset_id, exact_num_fields, min_fields, max_fields, list(eval(row['field_list'])), line_no=new_idx)
        if df is not None:
            yield df
        # update prev
        prev_locator = locator
        prev_idx = new_idx

def get_manyeyes_filtered_dfs(header_iter, exact_num_fields=None, min_fields=None, max_fields=None):
    corpus='manyeyes'
    base_dir = data_dirs[corpus]

    for next_line in header_iter:
        if next_line == 'EOF':
            return
        idx, row = next_line
        locator = row['locator']
        fields = eval(row['field_list']) #convert string to list
        full_file_path = join(base_dir, locator)
        dataset_id = locator.split('/')[-1]

        df = load_manyeyes(full_file_path, locator, dataset_id, exact_num_fields, min_fields, max_fields, fields)
        if df:
            yield df 
        else:
            continue


def get_plotly_filtered_dfs(header_iter, limit=None, exact_num_fields=None, min_fields=None, max_fields=None):
    corpus = 'plotly'
    base_dir = data_dirs[corpus]
    idx, row = next(header_iter)

    locator_buff = [row['locator']] # pending locators
    current_dataset_id = row['dataset_id']
    current_fields = list(eval(row['field_list']))

    #for f in locator_buff:
    while(len(locator_buff)>0):
        f = locator_buff.pop(0)
        file_path = join(base_dir, f)
        plotly_row_iter = plotly_file_iter(file_path)


        for chart_num, chart_obj in plotly_row_iter:

            if chart_obj.fid == current_dataset_id:
                # if matches the current dataset_id
                df = extract_plotly(chart_obj.table_data, f, chart_obj.fid, exact_num_fields, min_fields, max_fields, current_fields)
                if df is not None:
                    yield df

                # proceed to the next header after a fid match
                next_header_line = next(header_iter)

                if next_header_line != 'EOF':
                    idx, row = next_header_line
                    current_locator = row['locator']
                    current_dataset_id = row['dataset_id']
                    current_fields = list(eval(row['field_list']))
                    if current_locator != f:
                        locator_buff.append(current_locator)
                        break
                else:
                    return



get_dfs_by_corpus = {
    'plotly': get_plotly_dfs,
    'manyeyes': get_manyeyes_dfs,
    'webtables': get_webtables_dfs,
    'opendata': get_opendata_dfs
}

get_filtered_dfs_by_corpus = {
    'plotly': get_plotly_filtered_dfs,
    'manyeyes': get_manyeyes_filtered_dfs,
    'webtables': get_webtables_filterd_dfs,
    'opendata': get_opendata_filtered_dfs
}
