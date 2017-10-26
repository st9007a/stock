import datetime
import pickle
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from util.dataset import load, get_batch

start = '2016-06-30'
end = '2017-10-10'
pred_date = '2017-10-15'

d = start
i = 0

x_total = []
y_total = []

print('load data')
while(True):
    i += 5
    d = (datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
    if (datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.datetime.strptime(d, '%Y-%m-%d')).total_seconds() < 0:
        break

    x, y, _ =load(d)
    x_total += x
    y_total += y

print('load data complete')
print(np.array(x_total).shape)
print(np.array(y_total).shape)

cnt1 = 0
cnt2 = 0
cnt3 = 0

for y in y_total:
    if y == 1:
        cnt1 += 1
    elif y == -1:
        cnt2 += 1
    elif y == 0:
        cnt3 += 1

print('1: ' + str(cnt1))
print('-1: ' + str(cnt2))
print('0: ' + str(cnt3))

x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size = 0.2, random_state = 42)

clf = Pipeline([
               ('svd', TruncatedSVD(n_components = 20)),
               ('clf', RandomForestClassifier(n_estimators = 20, max_depth = 10, random_state = 0))
               ])

clf.fit(x_train, y_train)
print('training accuracy: ' + str(clf.score(x_train, y_train)))
print('validate accuracy: ' + str(clf.score(x_test, y_test)))
print('cross validation: ' + str(cross_val_score(clf, x_total, y_total, cv = 5)))

x_pred, y_pred, com = load(pred_date)

print('predict accuracy: ' + str(clf.score(x_pred, y_pred)))
pred = clf.predict(x_pred)

cnt1 = 0
cnt2 = 0
cnt3 = 0
for p in pred:
    if p == 1:
        cnt1 += 1
    elif p == -1:
        cnt2 += 1
    elif p == 0:
        cnt3 += 1

print('1: ' + str(cnt1))
print('-1: ' + str(cnt2))
print('0: ' + str(cnt3))

print('save model to `model.pkl`')
with open('./model.pkl', 'wb')as p:
    pickle.dump(clf, p, protocol = pickle.HIGHEST_PROTOCOL)
