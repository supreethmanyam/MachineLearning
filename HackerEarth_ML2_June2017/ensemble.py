import pandas as pd

xgb_count = pd.read_csv('../submissions/submit_xgb_count_prob.csv')
xgb_tfidf = pd.read_csv('../submissions/submit_xgb_tfidf_prob.csv')
lgbm_count = pd.read_csv('../submissions/submit_lgbm_count_prob.csv')
lgbm_tfidf = pd.read_csv('../submissions/submit_lgbm_tfidf_prob.csv')

xgb_preds = xgb_count.merge(xgb_tfidf, on='project_id', how='left', suffixes=('_xgb_count', '_xgb_tfidf'))
lgbm_preds = lgbm_count.merge(lgbm_tfidf, on='project_id', how='left', suffixes=('_lgbm_count', '_lgbm_tfidf'))

ens = xgb_preds.merge(lgbm_preds, on='project_id', how='left')

ens['final_status'] = ((ens['final_status_xgb_count']*0.25 + ens['final_status_xgb_tfidf']*0.25 + ens['final_status_lgbm_tfidf']*0.5)*0.3 + ens['final_status_lgbm_count']*0.7)

ens.loc[ens['final_status'] > 0.297, 'final_status'] = 1
ens.loc[ens['final_status'] <= 0.297, 'final_status'] = 0
ens['final_status'] = ens['final_status'].astype(int)
print(ens['final_status'].value_counts())
ens[['project_id', 'final_status']].to_csv('../submissions/ML2_SupreethManyam.csv', index=False)
