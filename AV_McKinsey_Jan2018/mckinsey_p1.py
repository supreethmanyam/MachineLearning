import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
import re
import os

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

PATH = 'data/'
submissions_path = 'submissions/'

train = pd.read_csv(f"{PATH}train.csv")
test = pd.read_csv(f"{PATH}test.csv")


def replace_lower_freq(df, col, replace_val, threshold):
    value_counts = df[col].value_counts()
    to_remove = value_counts[value_counts <= threshold].index
    return df[col].replace(to_remove, replace_val)


def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
              'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt, n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop:
        df.drop(fldname, axis=1, inplace=True)


def preprocess(df, id_col, target_col):
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    df['City_Code'] = replace_lower_freq(df, 'City_Code', "C00000", 5)
    df['City_Code'] = df['City_Code'].fillna("C00000")
    df['City_Code'] = df['City_Code'].str.lstrip('C').astype(int)

    df['City_Category'] = df['City_Category'].fillna("M")
    df['City_Category'] = df['City_Category'].map({'A': 1, 'B': 2, 'C': 3, 'M': -1})

    df['Employer_Code'] = replace_lower_freq(df, 'Employer_Code', "COM0000000", 5)
    df['Employer_Code'] = df['Employer_Code'].fillna("COM0000000")
    df['Employer_Code'] = df['Employer_Code'].str.lstrip('COM').astype(int)

    df['Employer_Category1'] = df['Employer_Category1'].fillna("M")
    df['Employer_Category1'] = df['Employer_Category1'].map({'A': 1, 'B': 2, 'C': 3, 'M': -1})

    df['Employer_Category2'] = df['Employer_Category2'].fillna(-1)
    df['Employer_Category2'] = df['Employer_Category2'].astype(int)

    df['DOB'] = df['DOB'].fillna('11/01/82')
    df['DOB'] = pd.to_datetime(df['DOB'])
    df.loc[df['DOB'].dt.year > 2000, 'DOB'] = df.loc[df['DOB'].dt.year > 2000, 'DOB'].apply(lambda x: x-relativedelta(years=100))
    df['Lead_Creation_Date'] = pd.to_datetime(df['Lead_Creation_Date'])

    df['Age_at_lead_creation'] = (df['Lead_Creation_Date'] - df['DOB']) / np.timedelta64(1, 'D')

    add_datepart(df, 'DOB')
    add_datepart(df, 'Lead_Creation_Date')

    replace_lower_freq(df, 'Customer_Existing_Primary_Bank_Code', "B000", 5)
    df['Customer_Existing_Primary_Bank_Code'] = df['Customer_Existing_Primary_Bank_Code'].fillna("B000")
    df['Customer_Existing_Primary_Bank_Code'] = df['Customer_Existing_Primary_Bank_Code'].str.lstrip('B').astype(int)

    df['Primary_Bank_Type'] = df['Primary_Bank_Type'].fillna("M")
    df['Primary_Bank_Type'] = df['Primary_Bank_Type'].map({'P': 1, 'G': 2, 'M': -1})

    df['Contacted'] = df['Contacted'].map({'Y': 1, 'N': 0})

    df['Source'] = df['Source'].str.strip('S').astype(int)

    df['Source_Category'] = df['Source_Category'].map({'A': -1, 'B': 1, 'C': 2,
                                                       'D': 3, 'E': 4, 'F': 5, 'G': 6})

    df['Existing_EMI'] = df['Existing_EMI'].fillna(0)

    df.loc[df.Contacted == 0, ['Loan_Amount', 'Loan_Period', 'Interest_Rate', 'EMI']] = 0
    df.loc[df.Loan_Amount.isnull(), ['Loan_Amount', 'Loan_Period', 'Interest_Rate', 'EMI']] = 0

    df['Interest_Rate'] = df['Interest_Rate'].fillna(df['Interest_Rate'].mean())
    df['EMI'] = df['EMI'].fillna(df['EMI'].mean())

    ids = df[id_col]
    if target_col in df.columns:
        y = df[target_col]
        df = df.drop([id_col, target_col], axis=1)
        return ids, df, y
    else:
        df = df.drop([id_col], axis=1)
        return ids, df


train_ids, X_train, Y = preprocess(train.copy(), 'ID', 'Approved')
test_ids, X_test = preprocess(test.copy(), 'ID', 'Approved')

X_train['L_P'] = X_train['Loan_Amount']/X_train['Loan_Period']
X_test['L_P'] = X_test['Loan_Amount']/X_test['Loan_Period']

X_train['Total_EMI'] = X_train['Existing_EMI'] + X_train['EMI']
X_test['Total_EMI'] = X_test['Existing_EMI'] + X_test['EMI']

X_train['Income_left'] = X_train['Monthly_Income']/X_train['Total_EMI']
X_test['Income_left'] = X_test['Monthly_Income']/X_test['Total_EMI']

X_train['Normalized_Income'] = X_train['Monthly_Income']/X_train['Age_at_lead_creation']
X_test['Normalized_Income'] = X_test['Monthly_Income']/X_test['Age_at_lead_creation']

params = {'objective': 'binary:logistic',
          'booster': 'gbtree',
          'eval_metric': 'auc',
          'nthread': 16,
          'silent': 1,
          'max_depth': 3,
          'subsample': 0.9,
          'min_child_weight': 1,
          "colsample_bytree": 0.9,
          'eta': 0.05,
          'verbose_eval': True,
          'seed': 0}

n_fold = 5
scores = []
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2018)
for i, (build_index, val_index) in enumerate(skf.split(X_train, Y)):
    print(f"\nFold {i+1}")
    build, y_build = X_train.loc[build_index, :], Y[build_index]
    val, y_val = X_train.loc[val_index, :], Y[val_index]
    final_test = X_test.copy()

    avg_inc_by_city = (build.pivot_table(index=['City_Code', 'City_Category'], values='Monthly_Income', aggfunc=np.mean)
                            .reset_index(drop=False)
                            .rename(columns={'Monthly_Income': 'Average_Monthly_Income_by_City'}))

    build = build.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')
    val = val.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')

    avg_inc_by_source_var = (build.pivot_table(index=['Source', 'Source_Category', 'Var1'], values='Monthly_Income', aggfunc=np.mean)
                                  .reset_index(drop=False)
                                  .rename(columns={'Monthly_Income': 'Average_Monthly_Income_by_source_var'}))

    build = build.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')
    val = val.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')

    build['Diff_inc_in_source_var'] = build['Monthly_Income'] - build['Average_Monthly_Income_by_source_var']
    val['Diff_inc_in_source_var'] = val['Monthly_Income'] - val['Average_Monthly_Income_by_source_var']

    final_test = final_test.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')
    final_test = final_test.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')
    final_test['Diff_inc_in_source_var'] = final_test['Monthly_Income'] - final_test['Average_Monthly_Income_by_source_var']

    dbuild = xgb.DMatrix(build, y_build)
    dval = xgb.DMatrix(val, y_val)
    dtest = xgb.DMatrix(final_test)

    num_rounds = 2000
    watchlist = [(dbuild, 'build'), (dval, 'val')]

    clf_xgb_fold = xgb.train(params, dbuild, num_rounds, evals=watchlist, early_stopping_rounds=50, verbose_eval=100)
    scores.append(roc_auc_score(y_val, clf_xgb_fold.predict(dval, ntree_limit=clf_xgb_fold.best_ntree_limit)))

    SUBMISSION = os.path.join(submissions_path, 'sub_simple_gbtree_{}.csv'.format(i))

    pred_test = clf_xgb_fold.predict(dtest, ntree_limit=clf_xgb_fold.best_ntree_limit)

    submission = pd.DataFrame({'ID': test_ids, 'Approved': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")
print(scores)
print(np.mean(scores))


params = {'objective': 'binary:logistic',
          'booster': 'dart',
          'eval_metric': 'auc',
          'nthread': 16,
          'silent': 1,
          'max_depth': 3,
          'subsample': 0.9,
          'min_child_weight': 1,
          "colsample_bytree": 0.9,
          'eta': 0.05,
          'verbose_eval': True,
          'seed': 0}

n_fold = 5
scores = []
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2018)
for i, (build_index, val_index) in enumerate(skf.split(X_train, Y)):
    print(f"\nFold {i+1}")
    build, y_build = X_train.loc[build_index, :], Y[build_index]
    val, y_val = X_train.loc[val_index, :], Y[val_index]
    final_test = X_test.copy()

    avg_inc_by_city = (build.pivot_table(index=['City_Code', 'City_Category'], values='Monthly_Income', aggfunc=np.mean)
                            .reset_index(drop=False)
                            .rename(columns={'Monthly_Income': 'Average_Monthly_Income_by_City'}))

    build = build.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')
    val = val.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')

    avg_inc_by_source_var = (build.pivot_table(index=['Source', 'Source_Category', 'Var1'], values='Monthly_Income', aggfunc=np.mean)
                                  .reset_index(drop=False)
                                  .rename(columns={'Monthly_Income': 'Average_Monthly_Income_by_source_var'}))

    build = build.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')
    val = val.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')

    build['Diff_inc_in_source_var'] = build['Monthly_Income'] - build['Average_Monthly_Income_by_source_var']
    val['Diff_inc_in_source_var'] = val['Monthly_Income'] - val['Average_Monthly_Income_by_source_var']

    final_test = final_test.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')
    final_test = final_test.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')
    final_test['Diff_inc_in_source_var'] = final_test['Monthly_Income'] - final_test['Average_Monthly_Income_by_source_var']

    dbuild = xgb.DMatrix(build, y_build)
    dval = xgb.DMatrix(val, y_val)
    dtest = xgb.DMatrix(final_test)

    num_rounds = 2000
    watchlist = [(dbuild, 'build'), (dval, 'val')]

    clf_xgb_fold = xgb.train(params, dbuild, num_rounds, evals=watchlist, early_stopping_rounds=50, verbose_eval=100)
    scores.append(roc_auc_score(y_val, clf_xgb_fold.predict(dval, ntree_limit=clf_xgb_fold.best_ntree_limit)))

    SUBMISSION = os.path.join(submissions_path, 'sub_simple_dart_{}.csv'.format(i))

    pred_test = clf_xgb_fold.predict(dtest, ntree_limit=clf_xgb_fold.best_ntree_limit)

    submission = pd.DataFrame({'ID': test_ids, 'Approved': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")
print(scores)
print(np.mean(scores))


clf_rf = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)

n_fold = 5
scores = []
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2018)
for i, (build_index, val_index) in enumerate(skf.split(X_train, Y)):
    print(f"\nFold {i+1}")
    build, y_build = X_train.loc[build_index, :], Y[build_index]
    val, y_val = X_train.loc[val_index, :], Y[val_index]
    final_test = X_test.copy()

    avg_inc_by_city = (build.pivot_table(index=['City_Code', 'City_Category'], values='Monthly_Income', aggfunc=np.mean)
                            .reset_index(drop=False)
                            .rename(columns={'Monthly_Income': 'Average_Monthly_Income_by_City'}))

    build = build.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')
    val = val.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')

    avg_inc_by_source_var = (build.pivot_table(index=['Source', 'Source_Category', 'Var1'], values='Monthly_Income', aggfunc=np.mean)
                                  .reset_index(drop=False)
                                  .rename(columns={'Monthly_Income': 'Average_Monthly_Income_by_source_var'}))

    build = build.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')
    val = val.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')

    build['Diff_inc_in_source_var'] = build['Monthly_Income'] - build['Average_Monthly_Income_by_source_var']
    val['Diff_inc_in_source_var'] = val['Monthly_Income'] - val['Average_Monthly_Income_by_source_var']

    final_test = final_test.merge(avg_inc_by_city, on=['City_Code', 'City_Category'], how='left')
    final_test = final_test.merge(avg_inc_by_source_var, on=['Source', 'Source_Category', 'Var1'], how='left')
    final_test['Diff_inc_in_source_var'] = final_test['Monthly_Income'] - final_test['Average_Monthly_Income_by_source_var']

    build = build.fillna(-1)
    val = val.fillna(-1)
    final_test = final_test.fillna(-1)
    build = build.replace([np.inf, -np.inf], -1)
    val = val.replace([np.inf, -np.inf], -1)
    final_test = final_test.replace([np.inf, -np.inf], -1)

    clf_rf.fit(build, y_build)
    scores.append(roc_auc_score(y_val, clf_rf.predict_proba(val)[:, 1]))

    SUBMISSION = os.path.join(submissions_path, 'sub_simple_rf_{}.csv'.format(i))

    pred_test = clf_rf.predict_proba(final_test)

    submission = pd.DataFrame({'ID': test_ids, 'Approved': pred_test[:, 1]})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")
print(scores)
print(np.mean(scores))
