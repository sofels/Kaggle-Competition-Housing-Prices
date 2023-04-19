# Kaggle Competition: Housing Prices - Advanced Regression Techniques

## Overview
This repo provides my submission to the [Kaggle House Prices - Advanced Regression Techniques Competition] (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).  The project involves developing regression models to predict housing prices.

## Files:
- Training Dataset: [train.csv](https://github.com/sofels/Kaggle-Competition-Housing-Prices/blob/main/train.csv)
- Test Dataset: [test.csv](https://github.com/sofels/Kaggle-Competition-Housing-Prices/blob/main/test.csv)
- Notebook: [Kaggle Housing Price Prediction.ipynb](https://github.com/sofels/Kaggle-Competition-Housing-Prices/blob/main/Kaggle%20Housing%20Price%20Prediction.ipynb)
- Submission: [Submission1.csv](https://github.com/sofels/Kaggle-Competition-Housing-Prices/blob/main/Kaggle%20Submission1.csv)

## Code Explanation
- Importing the data
- Data Inspection: Removal of columns with significant null values and evaluation both target and feature matrices
- Data Preprocessing: Replacement of null values and encoding of categorical data
- Comparing the models using cross-validation. Models used:
  - Linear Regression
  - RandomForestRegressor
  - GradientBoostingRegressor
  - XGBRegressor
  - SVR
- Hyperparameter tuning using GridSearchCV to find the better model
  - RandomForestRegressor
  - GradientBoostingRegressor
  - XGBRegressor
- Retraining the best model
- Exporting predictions for Kaggle competition
- Interpretation of Results
- Reflection


## Interpretation of Result
Four machine learning models were used: LinearRegression(LR), RandomForestRegressor (RFR), GradientBoostingRegressor(GBR) and XGBoostRegressor(XGBR). The training data initially had a shape of  (1460, 81). Features with significantly large number of null values were dropped. The resulting feature matrix was evaluated for correlation. The feature matrix was then preprocessed which increased the number of features to (1460, 237)  because of OneHotEncoder and OrdinalEncoder applied to the categorical data.

After preprocessing, cross validation was used to evaluate different models to identify the best performing model. The models used were LR, SVR, RFR, GBR and XGBR. The negative-root-mean-square (RMSE) was the metric used to evaluate the models. The RMSE for SVR and LR were very high with values of -80742190403579 and -80967 respectively. As a result, these models were not used to hyperparameter tuning. RFR, GBR and XGBR has RMSE values of -29810, -25986 and -3103 respectively.


Hyperparameters were tuned for the 3 selected models. Gridsearch was used in finding best model parameters. Learning_rate, max_depth and n_estimators were tuned for GBR. It was observed that increasing the n_estimators improved the RMSE with the best hyperparameters being learning_rate = 0.1, max_depth= 2, n_estimators=800 giving the best RMSE of -25119.91. This was further tuned with n_estimators=900 and resulted in better RMSE of -24565.71 for max_depth=2 and max_features= 0.7. 

For RFR, max_depth and max_features hyperparameters were tuned and values of max_depth=13 and max_features = 0.5 gave the best RMSE of -28162.51. Finally XGBR was applied to the dataset by tuning learning_rate and max_depth. The best RMSE of -27267.06 was obtained using learning_rate = 0.1 and max_depth = 3. 

Overall, GradientBoostingRegressor with n_estimators=900, max_depth=2 and max_features= 0.7 has the best RMSE of -24565.71. The final valiadtoin RMSE with the training dataset was 9516 with R^2 value of 0.986. A plot of the actual vs. predicted house prices showed that the house prices didnot deviate significantly from the unity line. Most of the points were clustered within 15000 of the unity line. 
