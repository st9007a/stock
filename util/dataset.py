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

    i = 0
    while (i < len(y)):
        for val in x[i]:
            b = False
            if val == -1:
                del x[i]
                del y[i]
                del com_list[i]
                i -= 1
                b = True
            if b == True:
                break
        i += 1

    # pad x
    for xi in x:
        if len(xi) < 30:
            xi += [0] * (30 - len(xi))

    # pprint(x)

    return x, y, com_list

def load_data(start, end):

    com_list = {}
    json_data_list = []
    i = 0
    while(True):
        i += 1
        d = (datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if (datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.datetime.strptime(d, '%Y-%m-%d')).total_seconds() < 0:
            break

        if os.path.isfile(src % (d)):
            json_data_list.append(load_json_file(d))

    for j in json_data_list:
        for com in j:
            if com == 'id' or com == 'taiex':
                continue

            is_jump = False
            for val in j[com]:
                if j[com][val] == 'NULL' or isinstance(j[com][val], str) == False:
                    is_jump = True
                    break

            if is_jump == True:
                continue

            if com not in com_list:
                com_list[com] = []

            com_list[com].append(j[com])

    x = []
    y = []
    for com in com_list:
        for i in range(len(com_list[com]) - 6):
            data = []
            for j in range(i, i + 5):
                data += [float(com_list[com][j][val]) for val in com_list[com][j]]

            x.append(data)

            if float(com_list[com][i + 5]['close']) - float(com_list[com][i + 4]['close']) > 0:
                y.append([1, 0])
                # y.append(1)
            else:
                y.append([0, 1])
                # y.append(0)

    return x, y

def get_pred(date):
    json_data_list = []
    com_list = {}
    i = 1
    while (len(json_data_list) < 20):
        d = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if os.path.isfile(src % (d)):
            json_data_list.append(load_json_file(d))
        i += 1

    json_data_list.reverse()

    for j in json_data_list:
        for com in j:
            if com == 'id' or com == 'taiex':
                continue

            is_jump = False
            for val in j[com]:
                if j[com][val] == 'NULL' or isinstance(j[com][val], str) == False:
                    is_jump = True
                    break

            if is_jump == True:
                continue

            if com not in com_list:
                com_list[com] = []

            com_list[com].append(j[com])

    x = []
    y = []

    for com in com_list:
        if len(com_list[com]) < 6:
            continue

        com_x = com_list[com][len(com_list[com]) - 6:len(com_list[com]) - 1]
        data = []
        for c in com_x:
            data += [float(c[val]) for val in c]

        x.append(data)

        if float(com_list[com][len(com_list[com]) - 1]['close']) - float(com_list[com][len(com_list[com]) - 2]['close']) > 0:
            y.append([1, 0])
            # y.append(1)
        else:
            y.append([0, 1])
            # y.append(0)

    print(np.array(x).shape)
    print(np.array(y).shape)

    return x, y

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
    i = 0
    while (i < len(x)):
        for val in x[i]:
            b = False
            if val == -1:
                del x[i]
                del com_list[i]
                i -= 1
                b = True
            if b == True:
                break
        i += 1

    return x, com_list

def get_full_data(start, end):
    x = []
    y = []
    i = 0
    while(True):
        i += 1
        d = (datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if (datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.datetime.strptime(d, '%Y-%m-%d')).total_seconds() < 0:
            break

        x_i, y_i, _ = load(d)
        x += x_i
        y += y_i

    y = [[1, 0] if i == 1 else [0, 1] for i in y]

    return x, y
