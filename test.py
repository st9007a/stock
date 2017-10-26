import numpy as np

from util.dataset import get_batch

for _ in range(100):
    x, y = get_batch(256)
    print('get_batch(256), except: (256, 42)')
    print('result: ' + str(np.array(x).shape))
