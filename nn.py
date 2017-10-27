import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

from util.dataset import load_data, pick_validate_data

model = Sequential()
model.add(Dense(1024, input_dim = 30))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

x_tr, y_tr = load_data('2017-07-01', '2017-10-20', onehot = True)
x_te, y_te = pick_validate_data('2017-10-25', onehot = True)

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

score = model.evaluate(x_te, y_te)

print('')
print('predict loss: ' + str(score[0]))
print('predict acc: ' + str(score[1]))
