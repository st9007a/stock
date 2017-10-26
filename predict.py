#!/usr/bin/python
import datetime
import json
import pickle
import sys
import numpy as np

from util.dataset import load_pred_data


date = sys.argv[1]

x, com_list = load_pred_data(date)
clf = None
with open('./model.pkl', 'rb')as p:
    clf = pickle.load(p)

pred = clf.predict(x)
disicion_list = []

i = 0
for com in com_list:
    if i >= 50:
        break

    if pred[i] == 0:
        i += 1
        continue

    type = None
    open_price = None
    if pred[i] == -1:
        type = 'short'
        open_price = x[i][1] + 0.1
    else:
        type = 'buy'
        open_price = x[i][1] - 0.1

    disicion = {
        'code': com,
        'type': type,
        'weight': 1,
        'life': 3,
        'open_price': open_price,
        'close_high_price': open_price + 0.15,
        'close_low_price': open_price - 0.15
    }

    disicion_list.append(disicion)
    i += 1

with open('../commit/' + date + '_' + date + '.json', 'w')as j:
    json.dump(disicion_list, j)
