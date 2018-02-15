
# [Analytics Vidhya - Click Prediction Hackathon](https://datahack.analyticsvidhya.com/contest/click-prediction/)
## Supreeth Manyam (ziron) Final submission

Scores 20.925 7th on the (Public leaderboard)[https://datahack.analyticsvidhya.com/contest/click-prediction/lb] and scores 19.942 4th on the (Private leaderboard)[https://datahack.analyticsvidhya.com/contest/click-prediction/pvt_lb]

### Platform: macOS High Sierra 10.13


### Packages



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Package</th>
      <th>Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>conda</td>
      <td>4.3.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>python</td>
      <td>3.6.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pypy</td>
      <td>Python 2.7.13-PyPy 5.8.0 with GCC 4.2.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>zip by Info-ZIP</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pandas</td>
      <td>0.20.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>numpy</td>
      <td>1.13.1</td>
    </tr>
  </tbody>
</table>
</div>



### Instructions to run
1. Place the train.csv and test.csv in "data" folder
2. Copy the data folder to the folder where all the code files are present.
3. Execute run.sh present in the code files using bash.

Note:
1. This creates a "submissions" folder which contains both csv and zip format of submission file when it is done.
2. Approximate run time is 30 minutes.

#### Initial tree structure
.<br>
├── FTRL_proximal.py<br>
├── Readme.md<br>
├── data<br>
│   ├── test.csv<br>
│   └── train.csv<br>
├── generate_submission.py<br>
├── run.sh<br>
└── train_and_predict.py<br>

1 directory, 7 files<br>

#### Final tree structure
.<br>
├── FTRL_proximal.py<br>
├── FTRL_proximal.pyc<br>
├── Readme.md<br>
├── data<br>
│   ├── test.csv<br>
│   ├── test_status.csv<br>
│   └── train.csv<br>
├── generate_submission.py<br>
├── run.sh<br>
├── submissions<br>
│   ├── ClickPrediction_ZIRON_Finalsubmission.csv<br>
│   └── ClickPrediction_ZIRON_Finalsubmission.zip<br>
└── train_and_predict.py<br>

2 directories, 11 files
