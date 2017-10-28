import json
import sys
import numpy as np

from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, LSTM

from util.dataset import load_data, pick_validate_data, load_pred_data

config = json.load(open('config.json', 'r'))

model = Sequential()
model.add(Dense(256, input_dim = 30))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

x_tr, y_tr = load_data(config['start'], config['end'], onehot = True)
x_te, y_te = pick_validate_data(config['eval'], onehot = True)

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

score = model.evaluate(x_te, y_te)

print('')
print('predict loss: ' + str(score[0]))
print('predict acc: ' + str(score[1]))

proba = model.predict(x_te)
stocks = []
final_x = []
y_f = []

i = 0
for p in proba:
    if (p[0] > config['up_lower_bound'] and p[0] < config['up_upper_bound']) or (p[1] > config['down_lower_bound'] and p[1] < config['down_upper_bound']):
        stocks.append(p)
        final_x.append(x_te[i])
        y_f.append(y_te[i])

    i += 1

total = 0
acc = 0
for s in stocks:
    if s[0] > 0.5 and y_f[total][0] == 1:
        acc += 1
    elif s[0] < 0.5 and y_f[total][1] == 1:
        acc += 1
    total += 1

print(str(acc) + '/' + str(total))
# model.save('nn_model.h5')
