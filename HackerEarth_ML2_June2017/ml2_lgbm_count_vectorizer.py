import pandas as pd
import numpy as np
import lightgbm as lgbm
from scipy import sparse
from datetime import datetime


def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
    return z


x_train = load_sparse_matrix('../input/train_count.npz')
x_test = load_sparse_matrix('../input/test_count.npz')
target = pd.read_csv('../input/train.csv', usecols=['final_status'])

train = pd.read_csv('../input/train.csv', usecols=['created_at', 'deadline', 'launched_at', 'state_changed_at'])
test = pd.read_csv('../input/test.csv', usecols=['created_at', 'deadline', 'launched_at', 'state_changed_at', 'project_id'])

train['created_at'] = pd.to_datetime(train['created_at'], unit='s')
train['state_changed_at'] = pd.to_datetime(train['state_changed_at'], unit='s')
train['deadline'] = pd.to_datetime(train['deadline'], unit='s')
train['launched_at'] = pd.to_datetime(train['launched_at'], unit='s')

test['created_at'] = pd.to_datetime(test['created_at'], unit='s')
test['state_changed_at'] = pd.to_datetime(test['state_changed_at'], unit='s')
test['deadline'] = pd.to_datetime(test['deadline'], unit='s')
test['launched_at'] = pd.to_datetime(test['launched_at'], unit='s')


def get_date_features(data, col):
    reference_date = datetime(2017, 6, 27, 22, 38, 59, 956695)
    day = data[col].dt.day
    week = data[col].dt.week
#     dayofweek = data[col].dt.dayofweek
    month = data[col].dt.month
    year = data[col].dt.year
    hour = data[col].dt.hour
    minutes = data[col].dt.minute
    seconds = data[col].dt.second
    age = (reference_date-data[col]).astype('timedelta64[h]')
    return np.vstack((seconds, minutes, hour, day, week, month, year, age)).T


c_tr = sparse.csr_matrix(get_date_features(train, 'created_at').astype(float))
s_tr = sparse.csr_matrix(get_date_features(train, 'state_changed_at').astype(float))
d_tr = sparse.csr_matrix(get_date_features(train, 'deadline').astype(float))
l_tr = sparse.csr_matrix(get_date_features(train, 'launched_at').astype(float))

c_te = sparse.csr_matrix(get_date_features(test, 'created_at').astype(float))
s_te = sparse.csr_matrix(get_date_features(test, 'state_changed_at').astype(float))
d_te = sparse.csr_matrix(get_date_features(test, 'deadline').astype(float))
l_te = sparse.csr_matrix(get_date_features(test, 'launched_at').astype(float))

x_train = sparse.hstack([x_train, c_tr, s_tr, d_tr, l_tr])
x_test = sparse.hstack([x_test, c_te, s_te, d_te, l_te])

ltrain = lgbm.Dataset(x_train, target.values.reshape(-1))
ltest = lgbm.Dataset(x_test)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'num_threads': 4,
    'seed': 2017,
    'bagging_seed': 2,
    'feature_fraction_seed': 256
}

num_rounds = 232
watchlist = [ltrain]
seeds = [127863, 125, 67498, 9864, 6578996, 24536, 82146, 50981]
i = 1
for s in seeds:
    print('Iteration-{}'.format(i))
    params['seed'] = s
    clf_lgbm_train = lgbm.train(params, ltrain, num_rounds, watchlist, verbose_eval=50)
    if i == 1:
        preds = clf_lgbm_train.predict(x_test)
    else:
        preds = preds + clf_lgbm_train.predict(x_test)
    i += 1
preds = preds/float(len(seeds))

submit = pd.DataFrame({'project_id': test['project_id'], 'final_status': preds})
submit[['project_id', 'final_status']].to_csv('../submissions/submit_lgbm_count_prob.csv', index=False)
