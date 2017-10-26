import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

from util.dataset import get_full_data

model = Sequential()
model.add(Dense(128, input_dim = 30))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam')

x_tr, y_tr = get_full_data('2017-01-01', '2017-10-01')
print(y_tr)

model.fit(x_tr, y_tr, epochs = 20, batch_size = 128, validation_split = 0.2, verbose = 1)

