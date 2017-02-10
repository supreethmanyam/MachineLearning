import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb


train = pd.read_csv('../input/train_63qYitG.csv')
test = pd.read_csv('../input/test_XaoFywY.csv')


train = train.fillna(-1)
test = test.fillna(-1)


train.loc[(0 < train['Trip_Distance']) & (train['Trip_Distance'] <= 10), 'Trip_Distance'] = 0
train.loc[(10 < train['Trip_Distance']) & (train['Trip_Distance'] <= 20), 'Trip_Distance'] = 1
train.loc[(20 < train['Trip_Distance']) & (train['Trip_Distance'] <= 30), 'Trip_Distance'] = 2
train.loc[(30 < train['Trip_Distance']) & (train['Trip_Distance'] <= 40), 'Trip_Distance'] = 3
train.loc[(40 < train['Trip_Distance']) & (train['Trip_Distance'] <= 50), 'Trip_Distance'] = 4
train.loc[(50 < train['Trip_Distance']) & (train['Trip_Distance'] <= 60), 'Trip_Distance'] = 5
train.loc[(60 < train['Trip_Distance']) & (train['Trip_Distance'] <= 70), 'Trip_Distance'] = 6
train.loc[(70 < train['Trip_Distance']) & (train['Trip_Distance'] <= 80), 'Trip_Distance'] = 7
train.loc[(80 < train['Trip_Distance']) & (train['Trip_Distance'] <= 90), 'Trip_Distance'] = 8
train.loc[(90 < train['Trip_Distance']) & (train['Trip_Distance'] <= 100), 'Trip_Distance'] = 9
train.loc[(100 < train['Trip_Distance']) & (train['Trip_Distance'] <= 110), 'Trip_Distance'] = 10
train.loc[(110 < train['Trip_Distance']) & (train['Trip_Distance'] <= 120), 'Trip_Distance'] = 11


test.loc[(0 < test['Trip_Distance']) & (test['Trip_Distance'] <= 10), 'Trip_Distance'] = 0
test.loc[(10 < test['Trip_Distance']) & (test['Trip_Distance'] <= 20), 'Trip_Distance'] = 1
test.loc[(20 < test['Trip_Distance']) & (test['Trip_Distance'] <= 30), 'Trip_Distance'] = 2
test.loc[(30 < test['Trip_Distance']) & (test['Trip_Distance'] <= 40), 'Trip_Distance'] = 3
test.loc[(40 < test['Trip_Distance']) & (test['Trip_Distance'] <= 50), 'Trip_Distance'] = 4
test.loc[(50 < test['Trip_Distance']) & (test['Trip_Distance'] <= 60), 'Trip_Distance'] = 5
test.loc[(60 < test['Trip_Distance']) & (test['Trip_Distance'] <= 70), 'Trip_Distance'] = 6
test.loc[(70 < test['Trip_Distance']) & (test['Trip_Distance'] <= 80), 'Trip_Distance'] = 7
test.loc[(80 < test['Trip_Distance']) & (test['Trip_Distance'] <= 90), 'Trip_Distance'] = 8
test.loc[(90 < test['Trip_Distance']) & (test['Trip_Distance'] <= 100), 'Trip_Distance'] = 9
test.loc[(100 < test['Trip_Distance']) & (test['Trip_Distance'] <= 110), 'Trip_Distance'] = 10
test.loc[(110 < test['Trip_Distance']) & (test['Trip_Distance'] <= 120), 'Trip_Distance'] = 11


train_id, test_id = train['Trip_ID'], test['Trip_ID']
target = train.Surge_Pricing_Type
train = train.drop(['Trip_ID', 'Surge_Pricing_Type'], axis=1)
test = test.drop(['Trip_ID'], axis=1)


target = target - 1


for f in train.columns:
    if train[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

dtrain = xgb.DMatrix(train, label=target)
dtest = xgb.DMatrix(test)


params = {'objective': 'multi:softprob',
          'booster': 'gbtree',
          'eval_metric': 'merror',
          'num_class': 3,
          'nthread': 4,
          'silent': 1,
          'max_depth': 6,
          'subsample': 0.9,
          'min_child_weight': 1,
          "colsample_bytree": 0.9,
          'eta': 0.1,
          'verbose_eval': True}


num_rounds = 170
for i in range(3):
    print i
    params['seed'] = i
    if i == 0:
        clf_xgb = xgb.train(params, dtrain, num_rounds)
        xgb_preds = clf_xgb.predict(dtest)
        final_preds = xgb_preds
    else:
        clf_xgb = xgb.train(params, dtrain, num_rounds)
        xgb_preds = clf_xgb.predict(dtest)
        final_preds = final_preds + xgb_preds


preds = final_preds/3

xgb_preds_arr = np.argmax(np.array(pd.DataFrame(preds).iloc[:, 0:3]), axis=1)


pred_xgb = pd.DataFrame({'Surge_Pricing_Type': xgb_preds_arr+1})
pred_xgb['Trip_ID'] = test_id
pred_xgb[['Trip_ID', 'Surge_Pricing_Type']].to_csv("../submissions/XGB.csv", index=False)
