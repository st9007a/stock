import json
import sys
import numpy as np

from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, LSTM

from util.dataset import load_data, pick_validate_data, load_pred_data
date = sys.argv[1]

model = Sequential()
model.add(Dense(256, input_dim = 30))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

x_tr, y_tr = load_data('2017-07-01', '2017-10-20', onehot = True)
# x_te, y_te = pick_validate_data('2017-10-25', onehot = True)
x_te, com = load_pred_data(date)

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

# score = model.evaluate(x_te, y_te)
#
# print('')
# print('predict loss: ' + str(score[0]))
# print('predict acc: ' + str(score[1]))

proba = model.predict(x_te)
stocks = []
final_com = []
final_x = []
# y_f = []

i = 0
for p in proba:
    if (p[0] > 0.5 and p[0] < 0.55) or (p[1] > 0.55 and p[1] < 0.6):
        stocks.append(p)
        final_com.append(com[i])
        final_x.append(x_te[i])
        # y_f.append(y_te[i])

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

with open('../commit/' + date + '_' + date + '.json', 'w')as j:
    json.dump(disicion_list, j)
# total = 0
# acc = 0
# for s in stocks:
#     if s[0] > 0.5 and y_f[total][0] == 1:
#         acc += 1
#     elif s[0] < 0.5 and y_f[total][1] == 1:
#         acc += 1
#     total += 1
#
# print(str(acc) + '/' + str(total))
# model.save('nn_model.h5')
