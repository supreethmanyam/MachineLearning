mkdir submissions
pypy train_and_predict.py
python generate_submission.py
zip submissions/ClickPrediction_ZIRON_Finalsubmission.zip submissions/ClickPrediction_ZIRON_Finalsubmission.csv -q
echo 'DONE!'
