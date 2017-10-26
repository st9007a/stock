import datetime
import json
import os
import random
import re
import sys

import numpy as np

from glob import glob
from pprint import pprint

exclude_dates = [
    '2004-02-11',
    '2004-02-12',
    '2004-02-13',
    '2004-02-16',
    '2004-02-17',
    '2004-02-18',
    '2004-02-19',
]

merry = '/home/db/stock_resource_center/resource/twse/json/%s.json'
luffy = '/home/mlb/res/stock/twse/json/%s.json'

src = merry
# load the json file via date
# in:
#   date { str }: date format string `YYYY-mm-dd`
# out:
#   data { dict }: data store json structire in file
def load_json_file(date):
    data = None
    with open(src % (date), 'r') as json_file:
        data = json.load(json_file)

    return data

# load data in the json file past week and parse them
# in:
#   date { str }: date format string `YYYY-mm-dd`, get json from this to past week
# out:
#   x { list }: 2-dimension list format training data, shape = (count of samples, 42)
#   y { list }: 1-dimension list format training data label, shape = (count of samples)
def load(date):
    tmp = []
    # f = []
    i = 0
    while (len(tmp) < 5):
        d = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if os.path.isfile(src % (d)):
            tmp.append(load_json_file(d))
            # f.append('./res/%s.json' % (d))
        i += 1

    com_stock = {}
    for com in tmp[0]:
        if com == 'id' or com == 'taiex':
            continue
        com_stock[com] = []

    for com in com_stock:
        for data in tmp:
            if com in data:
                com_stock[com].append(data[com])

    target_js_data = None
    for i in range(1, 100):
        d = (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if os.path.isfile(src % (d)):
            target_js_data = load_json_file(d)
            break

    x = []
    y = []
    com_list = []
    # print(f)
    for com in com_stock:
        if com not in target_js_data:
            continue

        com_list.append(com)

        _x = []
        for stock in com_stock[com]:
            for val in stock:
                # if val == 'volume':
                #     continue
                if re.match('^\d+\.\d+$', str(stock[val])) is not None or re.match('^\d+$', str(stock[val])) is not None:
                    _x.append(float(stock[val]))
                else:
                    _x.append(-1.)
        x.append(_x)


        if re.match('^\d+\.\d+$', str(target_js_data[com]['close'])) is None and re.match('^\d+$', str(target_js_data[com]['close'])) is None:
            y.append(-2)
        elif re.match('^\d+\.\d+$', str(com_stock[com][0]['close'])) is None and re.match('^\d+$', str(com_stock[com][0]['close'])) is None:
            y.append(-2)
        else:
            if float(target_js_data[com]['close']) - float(com_stock[com][0]['close']) > 0:
                y.append(1)
            elif float(target_js_data[com]['close']) - float(com_stock[com][0]['close']) < 0:
                y.append(-1)
            else:
                y.append(-2)

    i = 0
    while (i < len(y)):
        if y[i] == -2:
            del x[i]
            del y[i]
            del com_list[i]
            i -= 1
        i += 1

    # i = 0
    # while (i < len(y)):
    #     for val in x[i]:
    #         b = False
    #         if val == -1:
    #             del x[i]
    #             del y[i]
    #             del com_list[i]
    #             i -= 1
    #             b = True
    #         if b == True:
    #             break
    #     i += 1

    # pad x
    for xi in x:
        if len(xi) < 30:
            xi += [0] * (30 - len(xi))


    return x, y, com_list

# load the feature via date
# in:
#   date { str }: date formate string `YYYY-mm-dd`
# out:
#   x { list }: stock list with feature
def load_pred_data(date):
    tmp = []
    i = 1
    while (len(tmp) < 5):
        d = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        print(d)
        if os.path.isfile(src % (d)):
            tmp.append(load_json_file(d))
        i += 1

    com_stock = {}
    for com in tmp[0]:
        if com == 'id' or com == 'taiex':
            continue
        com_stock[com] = []

    for com in com_stock:
        for data in tmp:
            if com in data:
                com_stock[com].append(data[com])

    x = []
    com_list = []
    for com in com_stock:
        com_list.append(com)

        _x = []
        for stock in com_stock[com]:
            for val in stock:
                # if val == 'volume':
                #     continue
                if re.match('^\d+\.\d+$', str(stock[val])) is not None or re.match('^\d+$', str(stock[val])) is not None:
                    _x.append(float(stock[val]))
                else:
                    _x.append(-1.)
        x.append(_x)

    # pad x
    for xi in x:
        if len(xi) < 30:
            xi += [0] * (30 - len(xi))


    return x, com_list

# random select `batch_size` samples data
# in:
#   batch_size { int }: count of samples
# out:
#   x { list }: 2-dimension list, shape = (batch_size, 42)
#   y { list }: 1-dimemsion list, shape = (batch_size)
def get_batch(batch_size):
    if batch_size > 500:
        sys.exit('error: batch size is out of range (maximun size is 500)')

    file_list = glob('/home/db/stock_resource_center/resource/twse/json/*.json')
    date_list = [re.search('json\/(.+)\.json', file).group(1) for file in file_list]

    date = random.choice(date_list)
    while date in exclude_dates:
        date = random.choice(date_list)

    x, y, _ = load(date)
    l = len(y)
    p = random.randint(0, l - 1)

    x_batch = []
    y_batch = []
    for i in range(batch_size):
        x_batch.append(x[i % l])
        y_batch.append(y[i % l])

    return x_batch, y_batch
