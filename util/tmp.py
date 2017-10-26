import datetime
import os
import re

import numpy as np

def get_file_list(start, end):
    file_list = []
    s = datetime.datetime.strptime(start, '%Y-%m-%d')
    e = datetime.datetime.strptime(end, '%Y-%m-%d')

    # delta = int(str(e - s).split(' ')[0])

    for i in range(0, 7):
        date = (e - datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if os.path.isfile('./res/%s.json' % (date)):
            file_list.append('./res/%s.json' % (date))

    for j in range(1, 100):
        date = (e + datetime.timedelta(days = j)).strftime('%Y-%m-%d')
        if os.path.isfile('./res/%s.json' % (date)):
            target_file = './res/%s.json' % (date)

            return file_list, target_file

def get_data(raw_data, target_data):


def get_training_data(raw_data, target_data):
    x = [[] for _ in raw_data[0]]
    for data in raw_data:
        i = 0
        for stock in data:
            if stock == 'id' or stock == 'taiex':
                continue

            for val in data[stock]:
                if data[stock][val] == 'NULL':
                    x[i].append(-1.)
                else:
                    x[i].append(float(data[stock][val]))

            i += 1

    # i = 0
    # for data in x:
    #     if len(data) == 0:
    #         del x[i]
    #
    #     i += 1
    x.pop()
    x.pop()

    # padding 0 to let length == 5
    for _ in range(5 - len(raw_data)):
        for data in x:
            data += [0.] * 6

    print('x: ', np.array(x).shape)


    # generate label
    i = 0
    y = []
    for stock in target_data:
        if stock == 'id' or stock == 'taiex':
            continue

        if target_data[stock]['close'] == 'NULL':
            y.append(0)

        else:
            # label = float(target_data[stock]['close']) - x[i][1]
            # y.append(label)
            if float(target_data[stock]['close']) > x[i][1]:
                y.append(1)
            elif float(target_data[stock]['close']) < x[i][1]:
                y.append(-1)
            else:
                y.append(0)

        i += 1

    print('y: ', np.array(y).shape)

    return x, y
