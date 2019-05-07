import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import hstack
import string

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

translator = str.maketrans('', '', string.punctuation)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['desc'] = train['desc'].apply(lambda x: str(x).translate(translator).lower())
test['desc'] = test['desc'].apply(lambda x: str(x).translate(translator).lower())

train['keywords'] = train['keywords'].apply(lambda x: str(x.replace('-', ' ')).translate(translator).lower())
test['keywords'] = test['keywords'].apply(lambda x: str(x.replace('-', ' ')).translate(translator).lower())

count_vec = CountVectorizer(min_df=5, stop_words='english', max_features=5000)

train_vectors_desc = count_vec.fit_transform(train['desc'])
test_vectors_desc = count_vec.transform(test['desc'])

train_vectors_key = count_vec.fit_transform(train['keywords'])
test_vectors_key = count_vec.transform(test['keywords'])

train['created_at'] = pd.to_datetime(train['created_at'], unit='s')
train['state_changed_at'] = pd.to_datetime(train['state_changed_at'], unit='s')
train['deadline'] = pd.to_datetime(train['deadline'], unit='s')
train['launched_at'] = pd.to_datetime(train['launched_at'], unit='s')

test['created_at'] = pd.to_datetime(test['created_at'], unit='s')
test['state_changed_at'] = pd.to_datetime(test['state_changed_at'], unit='s')
test['deadline'] = pd.to_datetime(test['deadline'], unit='s')
test['launched_at'] = pd.to_datetime(test['launched_at'], unit='s')

train['c_s'] = (train['created_at'] - train['state_changed_at']).astype('timedelta64[h]')
train['s_d'] = (train['state_changed_at'] - train['deadline']).astype('timedelta64[h]')
train['d_l'] = (train['deadline'] - train['launched_at']).astype('timedelta64[h]')
train['l_c'] = (train['launched_at'] - train['created_at']).astype('timedelta64[h]')

test['c_s'] = (test['created_at'] - test['state_changed_at']).astype('timedelta64[h]')
test['s_d'] = (test['state_changed_at'] - test['deadline']).astype('timedelta64[h]')
test['d_l'] = (test['deadline'] - test['launched_at']).astype('timedelta64[h]')
test['l_c'] = (test['launched_at'] - test['created_at']).astype('timedelta64[h]')

cols_to_use = ['name', 'desc']
len_feats = ['name_len', 'desc_len']
count_feats = ['name_count', 'desc_count']

for i in np.arange(2):
    train[len_feats[i]] = train[cols_to_use[i]].apply(str).apply(len)
    train[count_feats[i]] = train[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))
train['keywords_len'] = train['keywords'].apply(str).apply(len)
train['keywords_count'] = train['keywords'].apply(str).apply(lambda x: len(x.split('-')))

for i in np.arange(2):
    test[len_feats[i]] = test[cols_to_use[i]].apply(str).apply(len)
    test[count_feats[i]] = test[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))
test['keywords_len'] = test['keywords'].apply(str).apply(len)
test['keywords_count'] = test['keywords'].apply(str).apply(lambda x: len(x.split('-')))

cols_for_model = np.setdiff1d(train.columns.values, ['project_id', 'name', 'desc', 'keywords',
                                                     'backers_count', 'final_status'])

for f in cols_for_model:
    if (train[f].dtype == 'object') or (train[f].dtype == 'datetime64[ns]'):
        print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

x_train = hstack([csr_matrix(train[list(cols_for_model)].astype(float).values), train_vectors_desc, train_vectors_key])
x_test = hstack([csr_matrix(test[list(cols_for_model)].astype(float).values), test_vectors_desc, test_vectors_key])

dtrain = xgb.DMatrix(x_train, label=train['final_status'])
dtest = xgb.DMatrix(x_test)

params = {'objective': 'binary:logistic',
          'booster': 'gbtree',
          'eval_metric': 'error',
          'nthread': 4,
          'silent': 1,
          'max_depth': 6,
          'subsample': 0.9,
          'min_child_weight': 1,
          "colsample_bytree": 0.9,
          'eta': 0.1,
          'seed': 2017}

num_rounds = 428
watchlist = [(dtrain, 'train')]
seeds = [127863, 125, 67498, 9864, 6578996, 24536, 82146, 50981]
i = 1
for s in seeds:
    print('Iteration-{}'.format(i))
    params['seed'] = s
    clf_xgb = xgb.train(params, dtrain, num_rounds, verbose_eval=50, evals=watchlist)
    if i == 1:
        preds = clf_xgb.predict(dtest)
    else:
        preds = preds + clf_xgb.predict(dtest)
    i += 1
preds = preds/float(len(seeds))

submit = pd.DataFrame({'project_id': test['project_id'], 'final_status': preds})
submit[['project_id', 'final_status']].to_csv('../submissions/submit_xgb_count_prob.csv', index=False)
# submit.loc[submit['final_status'] > 0.30, 'final_status'] = 1
# submit.loc[submit['final_status'] <= 0.30, 'final_status'] = 0
# submit['final_status'] = submit['final_status'].astype(int)
#
# submit[['project_id', 'final_status']].to_csv('../submissions/submit_xgb_count_sub.csv', index=False)

np.savez('../input/train_count', row=x_train.row, col=x_train.col,
         data=x_train.data, shape=x_train.shape)
np.savez('../input/test_count', row=x_test.row, col=x_test.col,
         data=x_test.data, shape=x_test.shape)
