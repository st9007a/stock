import json
import sys
import numpy as np

from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, LSTM

from util.dataset import load_data, pick_validate_data, load_pred_data

config = json.load(open('config.json', 'r'))

date = sys.argv[1]

model = Sequential()
model.add(Dense(256, input_dim = 30))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

x_tr, y_tr = load_data(config['start'], config['end'], onehot = True)
x_te, com = load_pred_data(date)

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

proba = model.predict(x_te)
stocks = []
final_com = []
final_x = []

i = 0
for p in proba:
    if (p[0] > config['up_lower_bound'] and p[0] < config['up_upper_bound']) or (p[1] > config['down_lower_bound'] and p[1] < config['down_upper_bound']):
        stocks.append(p)
        final_com.append(com[i])
        final_x.append(x_te[i])

    i += 1

disicion_list = []
i = 0
for xi in final_x:
    type = None
    if stocks[i][0] > stocks[i][1]:
        type = 'buy'
    else:
        type = 'short'

    disicion = {
        'code': final_com[i],
        'type': type,
        'weigth': 1,
        'life': 3,
        'open_price': 'open',
        'close_high_price': xi[1] + 0.1,
        'close_low_price': xi[1] - 0.1
    }
    disicion_list.append(disicion)
    i += 1

with open('../commit/' + date + '_' + date + '.json', 'w')as j:
    json.dump(disicion_list, j)
# model.save('nn_model.h5')
