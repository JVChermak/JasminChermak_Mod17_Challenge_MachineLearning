# Supervised Machine Learning

## Overview of the Project
Train and evaluate machine learning models to predict risky loans given data with unbalanced classes. The following resampling methods were employed:
1. Oversample data using RandomOverSampler
2. Oversample data using SMOTE
3. Undersample data using ClusterCentroids
4. Use a combination method using SMOTEENN

## Resources
- Data Source: LoanStats_2019Q1.csv
- Software: Python 3.7.1, Jupyter Notebook, Scikit-learn and Imbalanced-learn libraries

## Resampling Analysis
There were a number of categorical features in the original data set. To prepare to run the various machine learning models, pd.get_dummies() was used in the "encoding" version of each file with the first column dropped to remove redundancy. This was a change from the initial use of LabelEncoder because some items may have been given incorrect weight rather than being converted into binary data.

For the purpose of these models, loan status has been changed into 0 for high risk and 1 for low risk. In each model, the precision for predicting a high risk loan was terrible at 2% or 3%. The F1 score for high risk was also extremely low, with the highest for SMOTE having an F1 score of 6%, and the lowest for undersampling (ClusterCentroids) having an F1 of 3%. SMOTEENN had the highest balanced accuracy score of 79.73% and SMOTE being close behind with 79.66%. The highest recall score for the high risk loan status was the ClusterCentroid method with a score of 78%, but the recall of the low risk status was actually at least 10% lower than for the oversampling methods.

In the case of credit risk, it is better to look at the recall score with the idea that we know a certain number of loans are high risk and want to know how likely the model is to identify them. With that in mind, I would recommend the SMOTEENN method of resampling for the model. The F1 scores for SMOTE are 1% higher than SMOTEENN, but the recall score of SMOTEENN was 1% higher than SMOTE for the high risk loans. I would also suggest that the lending company be cautious in using this model because the precision rate for high risk loans is extremely low and therefore falsely identifies low risk loans as high risk loans.

## Extension
Train, evaluate and compare two different ensemble classifiers listed below to predict loan risk.
1. Balanced Random Forest Classifier
2. Easy Ensemble AdaBoost Classifier

## Ensemble Analysis
For the sake of comparison, pd.get_dummies() was also used for the ensemble models and loan status was changed so that 0 is high risk and 1 is low risk. While the precision of each model is still in the single digit percents, the recall of the Easy Ensemble AdaBoost Classifier, which correctly identifies high risk loans, is at 91%. This is the best prediction rate of any of the models and shows the least "false negatives" or falsely classifying a high risk loan as a low risk loan. The balanced accuracy score for Easy Ensemble was also much higher at 92.64%. The Easy Ensemble model is the better model to use. However, the lending company should be aware of the low precision rate for high risk loans at 8%, which falsely identifies several loans that are low risk as high risk.

To improve the Balanced Random Forest model, several of the features that have been identified as having 0 importance could be removed and the model run again.
