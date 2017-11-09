import datetime
import json
import sys
import numpy as np

from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, LSTM

from util.dataset import load_data, pick_validate_data, load_pred_data

def select_stock(model, date):
    x_te, com = load_pred_data(date)
    proba = model.predict(x_te)

    final_y = []
    final_com = []
    final_x = []

    i = 0
    for p in proba:
        if (p[0] > config['up_lower_bound'] and p[0] < config['up_upper_bound']) or (p[1] > config['down_lower_bound'] and p[1] < config['down_upper_bound']):
            final_y.append(p)
            final_com.append(com[i])
            final_x.append(x_te[i])

        i += 1

    return final_x, final_y, final_com

def gen_dis_file(final_x, final_y, final_com, date):
    disicion_list = []
    i = 0
    for xi in final_x:
        type = None
        if final_y[i][0] > final_y[i][1]:
            type = 'buy'
        else:
            type = 'short'

        disicion = {
            'code': final_com[i],
            'type': type,
            'weight': 1,
            'life': 3,
            'open_price': xi[1],
            'close_high_price': xi[1] + 0.1,
            'close_low_price': xi[1] - 0.1
        }
        disicion_list.append(disicion)
        i += 1

    with open('../commit/' + date + '_' + date + '.json', 'w')as j:
        json.dump(disicion_list, j)


config = json.load(open('config.json', 'r'))

date_from = sys.argv[1]
date_to = sys.argv[2]

model = Sequential()
model.add(Dense(256, input_dim = 30))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

x_tr, y_tr = load_data(config['start'], config['end'], onehot = True)

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

if date_from == date_to:
    final_x, final_y, final_com = select_stock(model, date_from)
    gen_dis_file(final_x, final_y, final_com, date_from)

else:
    i = 0
    disicion_file = {}
    while(True):
        d = (datetime.datetime.strptime(date_from, '%Y-%m-%d') + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        if (datetime.datetime.strptime(date_to, '%Y-%m-%d') - datetime.datetime.strptime(d, '%Y-%m-%d')).total_seconds() < 0:
            break

        i += 1

        final_x, final_y, final_com = select_stock(model, d)

        disicion_list = []
        j = 0
        for xi in final_x:
            type = None
            if final_y[j][0] > final_y[j][1]:
                type = 'buy'
            else:
                type = 'short'

            disicion = {
                'code': final_com[j],
                'type': type,
                'weight': 1,
                'life': 3,
                'open_price': xi[1],
                'close_high_price': xi[1] + 0.1,
                'close_low_price': xi[1] - 0.1
            }
            disicion_list.append(disicion)
            j += 1
        disicion_file[d] = disicion_list

    with open('../commit/' + date_from + '_' + date_to + '.json', 'w')as j:
        json.dump(disicion_file, j)

# model.save('nn_model.h5')
