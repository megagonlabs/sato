import re
import os
import base64
import hashlib
import pandas as pd

def canonical_header(h, max_header_len=30):
    # convert any header to its canonincal form
    # e.g. fileSize
    h = str(h)
    if len(h)> max_header_len:
        return '-'
    h = re.sub(r'\([^)]*\)', '', h) # trim content in parentheses
    h = re.sub(r"([A-Z][a-z])", r" \1", h) #insert a space before any Cpital starts
    words = list(filter(lambda x: len(x)>0, map(lambda x: x.lower(), re.split('\W', h))))
    if len(words)<=0:
        return '-'
    new_phrase = ''.join([words[0]] + [x.capitalize() for x in words[1:]])
    return new_phrase



def long_name_digest(name, n = 10):
    return base64.b32encode(hashlib.sha1(name.encode()).digest()[:n]).decode("utf-8") 



# Iterator for reading large header file.
def valid_header_iter_gen(file_name, CHUNK_SIZE=500):
    TYPENAME = os.environ['TYPENAME']

    valid_header_dir = os.path.join(os.environ['BASEPATH'], 'extract', 'out', 'headers', TYPENAME)

    valid_header_loc = os.path.join(valid_header_dir, file_name)    
    df_header = pd.read_csv(valid_header_loc, chunksize=CHUNK_SIZE)

    for chunk in df_header:
        for row in chunk.iterrows():
            yield row
    yield "EOF"

def count_length_gen(file_name):
    # count the numbers of lines in a file
    # Note: mannually -1 if there's header
    with open(file_name) as f:
        return sum(1 for line in f)

