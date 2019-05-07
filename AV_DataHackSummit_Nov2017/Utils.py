from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np


class preprocessing(object):

    def __init__(self):
        self.cols_for_prep = ['Loan_ID', 'Gender', 'Married', 'Education',
                              'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                              'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']
        self.cols_for_model = ['Loan_ID', 'Gender', 'Married', 'Education',
                               'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    def preprocess(self, data):
        data = data.loc[:, self.cols_for_prep]
        data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
        data['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
        data['Married'].fillna(data['Married'].mode()[0], inplace=True)
        data['Married'].replace({'Yes': 1, 'No': 0}, inplace=True)
        data['Education'].fillna(data['Education'].mode()[0], inplace=True)
        data['Education'].replace({'Graduate': 1, 'Not Graduate': 0}, inplace=True)
        data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
        data['Self_Employed'].replace({'Yes': 1, 'No': 0}, inplace=True)
        data['Credit_History'].fillna(0, inplace=True)
        data['ApplicantIncome'].fillna(data['ApplicantIncome'].mean(), inplace=True)
        data['CoapplicantIncome'].fillna(data['CoapplicantIncome'].mean(), inplace=True)
        data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
        data['Loan_Status'].replace({'Y': 1, 'N': 0}, inplace=True)
        return data[self.cols_for_prep]

    def split_data(self, data, test_size=50):
        X = data[self.cols_for_model]
        y = data['Loan_Status']
        train, test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2017, stratify=y)
        train_id = train['Loan_ID']
        test_id = test['Loan_ID']
        train = train.drop('Loan_ID', axis=1)
        test = test.drop('Loan_ID', axis=1)
        print('Train shape: {}'.format(train.shape))
        print('Test shape: {}'.format(test.shape))
        return train, test, y_train, y_test, train_id, test_id


class model:

    def __init__(self, train, y, test, id_train, id_test):
        self.train = train
        self.test = test
        self.y = y.values
        self.id_train = id_train.values
        self.id_test = id_test.values

    def fit_predict(self, clf):
        m = clf.fit(self.train, self.y)
        preds = m.predict(self.test)
        probs = m.predict_proba(self.test)
        return preds, probs[:, 1]

    def get_oof(self, clf, prediction_key):
        key = '{}_Prediction'.format(prediction_key)
        oof_train = pd.DataFrame({'Loan_ID': self.id_train, key: 0})
        oof_test = pd.DataFrame({'Loan_ID': self.id_test})
        test_preds = np.zeros(len(self.id_test))
        n_split = 5

        np.random.seed(2017)
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=123)
        X = self.train.reset_index(drop=True)
        for train_index, test_index in skf.split(X, self.y):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train = self.y[train_index]

            m = clf.fit(X_train, y_train)
            val_preds = m.predict_proba(X_val)[:, 1]
            oof_train.loc[test_index, key] = val_preds
            test_preds = test_preds + m.predict_proba(self.test)[:, 1]/n_split
        oof_test[key] = test_preds
        return oof_train, oof_test
