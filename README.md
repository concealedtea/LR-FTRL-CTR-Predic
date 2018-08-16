# <b> LR-FTRL-CTR-Predic </b>     
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)  
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/concealedtea/)

## Features
<b>Includes :</b>   
  
- Sorts through user data to determine features positive and negative to plot a logistic regression curve  
- Uses Google's [FTRL](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) stepping algorithm to better determine cluster amount for AdClick Prediction   

## Seperate Files and Explainations

<b>logistic_regression.py</b>   
```This sorts through the schema laid down to parse relevant columns, the ones that actually have data in them.```  
<b>logistic_regression_2.py</b>    
```This sorts through the entire | delimited list to find the AUC and LOGLOSS of the entire folder, I use this to compare in logloss_gain.py to determine which columns are detrimental and which are positive to the overall model.```  
<b>logloss_gain.py</b>    
```This sorts through the file with all the results in them to determine which features are positive & negative to our model.```  
<b>test_ftrl.py</b>    
```This uses Google's Follow-The-Regulated-Leader to adjust footstep length in logistic regression. Supposedly both faster and better than logistic regression.```
