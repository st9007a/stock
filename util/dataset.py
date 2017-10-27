import datetime
import json
import os
import random
import re
import sys

import numpy as np

from pprint import pprint

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

def convert2com_by_timeseries(json_data_list):
    com_list = {}
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

    return com_list

def load_data(start, end, onehot = False):
    json_data_list = []
    i = 0
    while(True):
        i += 1
        d = (datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if (datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.datetime.strptime(d, '%Y-%m-%d')).total_seconds() < 0:
            break

        if os.path.isfile(src % (d)):
            json_data_list.append(load_json_file(d))


    com_list = convert2com_by_timeseries(json_data_list)

    x = []
    y = []
    for com in com_list:
        for i in range(len(com_list[com]) - 6):
            data = []
            for j in range(i, i + 5):
                data += [float(com_list[com][j][val]) for val in com_list[com][j]]

            x.append(data)

            if float(com_list[com][i + 5]['close']) - float(com_list[com][i + 4]['close']) > 0:
                if onehot == True:
                    y.append([1, 0])
                else:
                    y.append(1)
            else:
                if onehot == True:
                    y.append([0, 1])
                else:
                    y.append(0)

    return x, y

def load_pred_data(date):
    json_data_list = []
    i = 1
    while (len(json_data_list) < 20):
        d = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if os.path.isfile(src % (d)):
            json_data_list.append(load_json_file(d))
        i += 1

    json_data_list.reverse()

    com_list = convert2com_by_timeseries(json_data_list)

    x = []
    coms = []

    for com in com_list:
        if len(com_list[com]) < 6:
            continue

        com_x = com_list[com][len(com_list[com]) - 6:len(com_list[com]) - 1]
        data = []
        for c in com_x:
            data += [float(c[val]) for val in c]

        x.append(data)
        coms.append(com)

    return x, coms

def pick_validate_data(date, onehot = False):
    json_data_list = []
    i = 1
    while (len(json_data_list) < 20):
        d = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if os.path.isfile(src % (d)):
            json_data_list.append(load_json_file(d))
        i += 1

    json_data_list.reverse()

    com_list = convert2com_by_timeseries(json_data_list)

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
            if onehot == True:
                y.append([1, 0])
            else:
                y.append(1)
        else:
            if onehot == True:
                y.append([0, 1])
            else:
                y.append(0)

    return x, y
