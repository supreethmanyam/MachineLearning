import pandas as pd
import numpy as np

lower_thresh = 0.007
upper_thresh = 0.03
print('############################### Reading generated predictions #############################\n')
test_status = pd.read_csv('data/test_status.csv', usecols=['ID', 'probability'])
all_test_ids = pd.read_csv('data/test.csv', usecols=['ID'], dtype={'ID': np.int32})

test_status['ConversionStatus'] = 0.0
test_status.loc[((test_status['probability'] >= lower_thresh) & (test_status['probability'] <= upper_thresh)),
                'ConversionStatus'] = 1.0
print('############################### Generate Payout #############################\n')
test_status['ConversionPayOut'] = 0.0
test_status.loc[test_status['ConversionStatus'] == 1.0, 'ConversionPayOut'] = 0.0015

print('{} is the number conversions'.format(test_status['ConversionStatus'].sum()))
print('{} is the sum of conversion payout'.format(test_status['ConversionPayOut'].sum()))

print('############################### Generate submission #############################\n')
partial_sub = pd.DataFrame({'ID': test_status['ID'], 'ConversionPayOut': test_status['ConversionPayOut']})
rest_sub = pd.DataFrame({'ID': all_test_ids.loc[~all_test_ids['ID'].isin(partial_sub['ID']), 'ID'],
                         'ConversionPayOut': 0})
submit = pd.concat((partial_sub, rest_sub))
print('Shape of submission: {}'.format(submit.shape))

print('############################### Writing to csv #############################\n')
submit[['ID', 'ConversionPayOut']].to_csv('submissions/ClickPrediction_ZIRON_Finalsubmission.csv', index=False)
print('############################### Writing csv to zip #############################\n')
