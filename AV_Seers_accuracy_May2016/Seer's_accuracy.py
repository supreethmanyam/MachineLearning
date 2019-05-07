import pandas as pd
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from sklearn import preprocessing
import xgboost as xgb
import copy
from sklearn.ensemble import RandomForestClassifier


seer = pd.read_csv("input/Train_seers_accuracy.csv", na_values=' ', parse_dates=[1,10], infer_datetime_format='%d-%b-%y')
seer_data = seer.copy(deep=True)
seer_data['dob_year'], seer_data['dob_month'], seer_data['dob_day'] = seer_data['DOB'].dt.year, seer_data['DOB'].dt.month, seer_data['DOB'].dt.day
seer_data['Tran_year'], seer_data['Tran_month'], seer_data['Tran_weekday'], seer_data['Tran_day'], seer_data['Tran_week']  = seer_data['Transaction_Date'].dt.year, seer_data['Transaction_Date'].dt.month, seer_data['Transaction_Date'].dt.weekday, seer_data['Transaction_Date'].dt.day, seer_data['Transaction_Date'].dt.week

def edit_year(x):
    if x > 2015:
        x = x - 100
        return x
    else:
        return x

seer_data['dob_year'] = seer_data['dob_year'].apply(edit_year)
seer_data['DOB'] = seer_data.dob_year.astype(str).str.cat(seer_data.dob_month.astype(str), sep='-').str.cat(seer_data.dob_day.astype(str), sep='-')

def cal_age(a):
    return relativedelta(datetime.datetime.now(), parse(a)).years
seer_data['Age'] = seer_data['DOB'].apply(cal_age)

preprocessed = pd.DataFrame()
preprocessed = seer_data[['Transaction_Date', 'Store_ID', 'Number_of_EMI',
       'Purchased_in_Sale', 'Var1', 'Var2', 'Var3', 'Client_ID', 'Gender',
       'Referred_Friend', 'Sales_Executive_ID',
       'Sales_Executive_Category', 'Lead_Source_Category', 'Payment_Mode',
       'Product_Category', 'Transaction_Amount', 'Tran_year', 'Tran_month', 'Tran_weekday',
       'Tran_day', 'Tran_week', 'Age']]

data_2003 = preprocessed[preprocessed.Tran_year == 2003]
data_2004 = preprocessed[preprocessed.Tran_year == 2004]
data_2005 = preprocessed[preprocessed.Tran_year == 2005]
data_2006 = preprocessed[preprocessed.Tran_year == 2006]


data_2003.loc[:, 'TARGET'] = 0
targets = pd.unique(pd.concat([data_2006.Client_ID], ignore_index=False))
data_2003.loc[data_2003.Client_ID.isin(targets), 'TARGET'] = 1

data_2004.loc[:, 'TARGET'] = 0
targets = pd.unique(pd.concat([data_2006.Client_ID], ignore_index=False))
data_2004.loc[data_2004.Client_ID.isin(targets), 'TARGET'] = 1

data_2005.loc[:, 'TARGET'] = 0
targets = pd.unique(pd.concat([data_2006.Client_ID], ignore_index=False))
data_2005.loc[data_2005.Client_ID.isin(targets), 'TARGET'] = 1


train = pd.concat([data_2003,data_2004,data_2005])
test = preprocessed.copy(deep=True)


train_id, test_id = train['Client_ID'], test['Client_ID']
target = train.TARGET
train = train.drop(['TARGET'], axis=1)

train = train.fillna(-1)
test = test.fillna(-1)


for f in train.columns:
    if train[f].dtype=='object' or 'datetime64[ns]':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

dtrain = xgb.DMatrix(train, label=target)
dtest = xgb.DMatrix(test)


params = {'objective': 'binary:logistic',
          'booster': 'gbtree',
          'eval_metric': 'auc',
          'nthread': 4,
          'silent': 1,
          'max_depth': 3,
          'subsample': 0.6,
          'min_child_weight': 1,
          "colsample_bytree": 0.4,
          'eta': 0.1,
          'verbose_eval': True,
          'seed': 0}

num_rounds = 130
watchlist = [(dtrain, 'train')]
clf_xgb = xgb.train(params, dtrain, num_rounds, verbose_eval=True, maximize=True, evals=watchlist)
xgb_preds = clf_xgb.predict(dtest)


pred_xgb = pd.DataFrame({'Cross_Sell': xgb_preds})
pred_xgb['Client_ID'] = test_id
pred_xgb = pred_xgb.groupby('Client_ID', as_index=False)['Cross_Sell'].mean()
pred_xgb[['Client_ID', 'Cross_Sell']].to_csv("Final_submission.csv", index=False)