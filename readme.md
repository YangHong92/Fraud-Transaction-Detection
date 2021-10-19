# Fraud Transaction Detection

## Project Overview
Predict on highly imbalanced test set (positive: negative: 0.016).

Metrics are Recall, Precision and F1 score. After evaluating model performance using those metrics, it shows that the feature engineering helps in improving models' performance in all metrics (Recall, Precision, F1 score). 

Candidate models are weighted Logistic Regression, Random Forest, LightGBM. It is found that the Precision is hard to improve, all above models demonstrated high Recall (around 0.75) but low Precision (around 0.04) and F1 score (0.08).

To further improve the model’s performance, model stacking and model cascading are implemented. The result shows that:
- By further stacking models, the meta-learner can achieve the highest Recall (0.772), with above average F1 score (0.082);
- By cascading models, the final output shows that the Precision is significantly improved by 5 times (0.218), and achieve the highest F1 score (0.207).

Generally speaking, this approach resolves the low precision issue observed in all previous models.

## Project Highlight
-	Given data distribution, proposed hypothesis while doing EDA
-	Discovered some interesting findings and based on which manually derive several features:
    -  Compare ‘cardCVV’ and ‘enteredCVV’ -> ‘isCVV_correct’
    - Compare ‘merchantCountryCode’ and ‘acqCountry’ -> ‘isDomensticTransaction’ 
    - Compare day difference between transactionDateTime and accountOpenDate -> days_after_signup
    - Group by customerID and cardLast4Digits -> numOfCards per customer 
    - Percentage calculation: balance/limit, transaction/balance, transaction/limit
    - Log-transform skewed features
    - Use K-means to assign cluster for each transaction
    - Include PCA first 3 components as it shows hyperplane separable for fraud and non-fraud transactions
    - Visualize ROC curve, PR curve and feature importance 

## Steps:
1.	Train test split (Stratified sampling)
2.	Data preprocess:
o	Numeric feature: standardisation
o	Categorical feature: target encoding, one hot encoding, binary encoding
3.	KNN imputer + Feature engineering
4.	Grid search + Class weight or Grid search + Down-sampling
5.	Performance improvement:
o	Model Stacking
o	Model Cascading

## Future Work:
-	Explore Datetime and derive new features
-	Up-sampling using SMOTE
