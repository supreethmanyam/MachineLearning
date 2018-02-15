# [McKinsey Analytics Online Hackathon Solution](https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-ii/)- Ziron

## Approach

Final model is an ensemble of (part:1 weighted average output of 1 XGB(gbtree), 1 XGB(dart) and 1 RFs) and (part:2 stacked output of 3 LGBs and 3 XGBs) on the same preprocessing steps. The heavy ensemble approach along with basic preprocessing steps and basic feature engineering helped in making stable predictions based on internal validation.

## Scores
[Public LB](https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-ii/lb) -- 0.8542547819 -- 38th position
[Private LB](https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-ii/pvt_lb) -- 0.8547508143 -- 3rd position 
Internal CV ~0.850

Note: I have uploaded code only for the first part of the models as the improvement is not significant with complete ensemble.

## Feature engineering
1. Date features (day, dayofweek, month, ...) for both DOB, Lead_Creation_Date
2. Age (Lead_Creation_Date - DOB)
3. Ratio of Loan amount and Loan period
4. Sum of Existing EMI and EMI for Loan
5. Ratio of Income and Total EMI
6. Ratio of Income and Age
7. OOF feature of average monthly income by city code and category
8. OOF feature of average monthly income by Source, source category and Var1(Anonymized feature)
9. Difference between Monthly income and the average income by source, source category and Var1

### Versions
```
Python 3.6.4 :: Anaconda custom (64-bit)

>>> pd.__version__
'0.22.0'
>>> np.__version__
'1.13.3'
>>> xgb.__version__
'0.7'
>>> lgb.__version__
'2.0.12'
>>> sk.__version__
'0.19.1'
```
### Machine specifications
Ubuntu 16.04.3 LTS, 16 vCPUs, 14.4 GB memory

PS: I accidentally included Year in the date features which is mistake. Thankfully tree based models handles this by itself.
