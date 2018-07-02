CRT-Predictor-LL-FTLR

Seperate Files and Explainations

logistic_regression.py  
This sorts through the schema laid down to parse relevant columns, the ones that actually have data in them.
logistic_regression_2.py  
This sorts through the entire | delimited list to find the AUC and LOGLOSS of the entire folder, I use this to compare in logloss_gain.py to determine which columns are detrimental and which are positive to the overall model.
logloss_gain.py  
This sorts through the file with all the results in them to determine which features are positive & negative to our model.
test_ftrl.py  
This uses Google's Follow-The-Regulated-Leader to adjust footstep length in logistic regression. Supposedly both faster and better than logistic regression.
