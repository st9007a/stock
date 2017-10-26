import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

from util.dataset import load_data, get_pred

model = Sequential()
model.add(Dense(256, input_dim = 30))
model.add(Activation('relu'))
model.add(BatchNormalization())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

x_tr, y_tr = load_data('2017-07-01', '2017-10-01')
x_te, y_te = get_pred('2017-10-15')

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

score = model.evaluate(x_te, y_te)

print('')
print('predict loss: ' + str(score[0]))
print('predict acc: ' + str(score[1]))
