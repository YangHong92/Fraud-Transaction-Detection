# Fraud Transaction Detection

## Project Overview
Predict on highly imbalanced test set (positive/negative ratio: 0.016).

### Metrics
Recall, Precision, F1 score

### Candidate models
Logistic Regression, Random Forest, LightGBM

### Steps
1.	Train baseline models on data without feature engineering
2.	Train test split (stratified sampling)
3.	Data preprocess:
	- Numeric feature: standardisation
	- Categorical feature: target encoding, one hot encoding, binary encoding
4.	Feature engineering
5.	Grid search + Class weight or Down-sampling
6.	Performance improvement:
	- Model Stacking
	- Model Cascading

### Result
After feature engineering, models' performances are observed improvement in all metrics (Recall, Precision, F1 score) compared with baseline models. 

It is found that the Precision is hard to improve, all above models demonstrated high Recall (around 0.75) but low Precision (around 0.04) and F1 score (0.08).

To further improve the model’s performance, model stacking and model cascading are implemented:
- By further stacking models, the meta-learner can achieve the highest Recall (0.772), with above average F1 score (0.082);

    The architecture of model stacking:
![stack model architecture](https://github.com/YangHong92/Fraud-Transaction-Detection/raw/master/stacking_model_architecture.png)

- By cascading models, the final output shows that the Precision is significantly improved by 5 times (0.218), and achieves the highest F1 score (0.207). Generally speaking, this approach solves the low precision issue observed in all previous models.

    The architecture of model cascading:
![cascade model architecture](https://github.com/YangHong92/Fraud-Transaction-Detection/raw/master/cascade_model_architecture.png)

## Project Highlight
- Discovered some interesting findings and based on which manually derive several features:
    - Compare ‘cardCVV’ and ‘enteredCVV’ -> ‘isCVV_correct’
    - Compare ‘merchantCountryCode’ and ‘acqCountry’ -> ‘isDomensticTransaction’ 
    - Compare day difference between transactionDateTime and accountOpenDate -> days_after_signup
    - Group by customerID and cardLast4Digits -> numOfCards per customer 
    - Percentage calculation: balance/limit, transaction/balance, transaction/limit
    - Log-transform skewed features
    - Use K-means to assign cluster for each transaction
    - Include PCA first 3 components as it shows hyperplane separable for fraud and non-fraud transactions
- Visualize the ROC curve, PR curve and feature importance for trained models
- Applied Model Stacking
- Applied Model Cascading

## Future Work
-	Explore Datetime and derive new features
-	Up-sampling using SMOTE
