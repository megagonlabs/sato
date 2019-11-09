# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc
from scipy.stats import entropy, normaltest, mode
import pandas as pd

import re
import ast
from decimal import Decimal
from random import sample

from collections import OrderedDict

import datetime
import dateutil.parser as dparser
from .dateparser import DATE_FORMATS, is_date

general_types = [ 'c', 'q', 't' ]

# Type Detection Helpers
data_types = [
    'id',
    'string',
    'boolean',
    # 'percentage',
    'currency',
    'integer',
    'decimal',
    'time',
    'date',
    'month',
    'year',
]

data_type_to_general_type = {
    'id': 'c',
    'string': 'c',
    'boolean': 'c',
    'percentage': 'q',
    'currency': 'q',
    'integer': 'q',
    'decimal': 'q',
    'time': 't',
    'date': 't',
    'month': 't',
    'year': 't'
}

locales = [
    'en_US.UTF8',
    'es_ES.UTF8'
]

replace_strings = (',', '%', '$', '€', '"', "'")
def replace_special_characters_in_numeric(v):
    result = []
    for e in v:
        for r in replace_strings:
            e = str(e).replace(r, '')
        result.append(e)
    return result

# https://stackoverflow.com/questions/2811031/decimal-or-numeric-values-in-regular-expression-validation
integer_regex = re.compile("^-?(0|[1-9]\d*)(?<!-0)$")
decimal_regex = re.compile("^(?!-0?(\.0+)?$)-?(0|[1-9]\d*)?(\.\d+)?(?<=\d)$")
    
def detect_string(e):   
    return True
    
def detect_integer(e):
    if e == '' or pd.isnull(e): return False

    try:
        if integer_regex.match(e): return True
    except:
        try:
            if float(e).is_integer(): return True
        except:
            try:
                for l in locales:
                    locale.setlocale(locale.LC_all, l)
                    if float(locale.atoi(e)).is_integer(): return True
            except:
                pass
    return False
    
def detect_decimal(e):
    if e == '' or pd.isnull(e): return False

    if decimal_regex.match(e):
        return True
    try:
        d = Decimal(e)
        return True
    except:
        try:
            for l in locales:
                locale.setlocale(locale.LC_all, l)          
                value = locale.atof(e)
                if sys.version_info < (2, 7):
                    value = str(e)
                return Decimal(e)
        except:
            pass
    return False

def detect_percentage(e):
    return

true_values = ('yes', 'y', 'true', 't', '0')
false_values = ('no', 'n', 'false', 'f', '1')
boolean_values = true_values + false_values
def detect_boolean(e):
    if e.strip().lower() in boolean_values:
        return True
    return False

def detect_year(e):
    if len(e) >= 4 and e.startswith(('18', '19', '20', '21')):
        return True
    return False

months = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_strings = months + [ s.lower() for s in months] + [ s[:3] for s in months ] + [ s[:3].lower() for s in months]
def detect_month(e):
    if e in month_strings:
        return True
    return False

def detect_time(e):
    if isinstance(e, datetime.time):
        return True
    return False
 
def detect_date(e):
    if is_date(e): return True
    for date_type in [ datetime.datetime, datetime.date, np.datetime64 ]:
        if isinstance(e, date_type): return True

    # Slow!!!
    # for date_format in DATE_FORMATS:
    #     try:
    #         if datetime.strptime(e, date_format):
    #             return True
    #     except:
    #         continue

    # Also slow
    # try: 
    #   dparser.parse(e)
    #   return True
    # except: pass
    return False
    
type_weights = {
    'string': 1,
    'boolean': 7,
    'integer': 6,
    'decimal': 4,
    'time': 6,
    'date': 6,
    'month': 7,
    'year': 7
}

data_type_to_function = {
    'string': detect_string,
    'boolean': detect_boolean,  
    'integer': detect_integer,
    'decimal': detect_decimal,
    'time': detect_time,
    'date': detect_date,
    'month': detect_month,
    'year': detect_year
}
    
def detect_field_type(field_name, v, num_samples=100):
    type_scores = {
        'string': 1,
        'boolean': 0,
        'integer': 0,
        'decimal': 0,
        'time': 0,
        'date': 0,
        'month': 0,
        'year': 0
    }

    date_strings = ('date', 'Date', 'DATE')
    if field_name in date_strings or field_name.endswith(date_strings):
        return 'date', {}

    month_strings = ('month', 'Month', 'MONTH')
    if field_name in month_strings or field_name.endswith(month_strings):
        return 'month', {}

    year_strings = ('year', 'Year', 'YEAR')
    if field_name in year_strings or field_name.endswith(year_strings):
        return 'year', {}

    id_strings = ('id', 'ID', 'Id', 'name')
    if field_name in id_strings or field_name.endswith(id_strings):
        return 'id', {}

    v = sample(v, min(len(v), num_samples))

    for e in v:
        if e is None or e == '' or pd.isnull(e): continue

        e = str(e)

        for r in replace_strings:
            e = e.replace(r, '')

        for data_type, detection_function in data_type_to_function.items():
            if detection_function(e): type_scores[data_type] += type_weights[data_type]

    score_tuples = []
    for type_name, score in type_scores.items():
        score_tuples.append([ type_name, score ])
    final_field_type = max(score_tuples, key=lambda t: t[1])[0]

    return final_field_type, score_tuples