Loan Default Prediction Using XGBoost (UCI Credit Card Dataset)

This repository contains an end-to-end machine learning project using the "Default of Credit Card Clients Dataset" from the UCI Machine Learning Repository. The objective is to predict whether a credit card client will default on their payment next month using historical financial and demographic data.

---------------------------------------------------------
Dataset
---------------------------------------------------------
- Source: UCI Machine Learning Repository
- File Name: default of credit card clients.xls
- Rows: 30,000
- Columns: 24
- Target Variable: 'default payment next month' (renamed to 'default')
  - 1 → client will default next month
  - 0 → client will not default

---------------------------------------------------------
Libraries Used
---------------------------------------------------------
- numpy
- pandas
- scikit-learn
- xgboost

To install:
pip install numpy pandas scikit-learn xgboost

---------------------------------------------------------
Steps Followed
---------------------------------------------------------
1. Load the Dataset
- Loaded with pandas.read_excel() using header=1

2. Preprocessing
- Dropped the 'ID' column
- Renamed 'default payment next month' to 'default'
- Separated features (X) and target (y)

3. Train-Test Split
- 80/20 split using train_test_split
- Stratified by target variable for balance

4. Feature Scaling
- Standardized features using StandardScaler

5. Model Training: XGBoost with GridSearchCV
- Tuned parameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- Used 3-fold cross-validation and accuracy as the scoring metric

6. Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report

---------------------------------------------------------
Final Results
---------------------------------------------------------
Best Parameters:
{'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}

Accuracy: 0.8180

Confusion Matrix:
[[4435  238]
 [ 854  473]]

Classification Report:
              precision    recall  f1-score   support
           0       0.84      0.95      0.89      4673
           1       0.67      0.36      0.46      1327
    accuracy                           0.82      6000

---------------------------------------------------------
Author
---------------------------------------------------------
Pratyush Raman  
Email: ramanpratyush101@gmail.com  
LinkedIn: https://www.linkedin.com/in/pratyushraman10/
