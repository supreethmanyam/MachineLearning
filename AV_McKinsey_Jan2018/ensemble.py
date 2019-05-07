import pandas as pd
import numpy as np
import os

submissions_path = 'submissions/'


stacked_1 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_gbtree_0.csv'))
stacked_2 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_gbtree_1.csv'))
stacked_3 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_gbtree_2.csv'))
stacked_4 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_gbtree_3.csv'))
stacked_5 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_gbtree_4.csv'))
sub_gbtree = pd.DataFrame()
sub_gbtree['ID'] = stacked_1['ID']
sub_gbtree['Approved'] = np.exp(np.mean(
    [
        stacked_1['Approved'].apply(lambda x: np.log(x)),
        stacked_2['Approved'].apply(lambda x: np.log(x)),
        stacked_3['Approved'].apply(lambda x: np.log(x)),
        stacked_4['Approved'].apply(lambda x: np.log(x)),
        stacked_5['Approved'].apply(lambda x: np.log(x)),
        ], axis=0))

stacked_1 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_dart_0.csv'))
stacked_2 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_dart_1.csv'))
stacked_3 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_dart_2.csv'))
stacked_4 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_dart_3.csv'))
stacked_5 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_dart_4.csv'))
sub_dart = pd.DataFrame()
sub_dart['ID'] = stacked_1['ID']
sub_dart['Approved'] = np.exp(np.mean(
    [
        stacked_1['Approved'].apply(lambda x: np.log(x)),
        stacked_2['Approved'].apply(lambda x: np.log(x)),
        stacked_3['Approved'].apply(lambda x: np.log(x)),
        stacked_4['Approved'].apply(lambda x: np.log(x)),
        stacked_5['Approved'].apply(lambda x: np.log(x)),
        ], axis=0))

stacked_1 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_rf_0.csv'))
stacked_2 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_rf_1.csv'))
stacked_3 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_rf_2.csv'))
stacked_4 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_rf_3.csv'))
stacked_5 = pd.read_csv(os.path.join(submissions_path, 'sub_simple_rf_4.csv'))
sub_rf = pd.DataFrame()
sub_rf['ID'] = stacked_1['ID']
sub_rf['Approved'] = np.exp(np.mean(
    [
        stacked_1['Approved'].apply(lambda x: np.log(x)),
        stacked_2['Approved'].apply(lambda x: np.log(x)),
        stacked_3['Approved'].apply(lambda x: np.log(x)),
        stacked_4['Approved'].apply(lambda x: np.log(x)),
        stacked_5['Approved'].apply(lambda x: np.log(x)),
        ], axis=0))

sub = pd.DataFrame()
sub['ID'] = sub_gbtree['ID']
sub['Approved'] = np.exp(np.mean(
    [
        sub_dart['Approved'].apply(lambda x: np.log(x)),
        sub_gbtree['Approved'].apply(lambda x: np.log(x)),
        ], axis=0))
sub['Approved'] = sub['Approved']*0.85 + sub_rf['Approved']*0.15
sub[['ID', 'Approved']].to_csv('FinalSubmission_Ziron.csv', index=False, float_format='%.6f')
