# Telco Customer Churn Prediction

## Project Overview
An end-to-end Machine Learning pipeline designed to predict customer churn in a telecommunications company. The objective is to identify high-risk customers who are likely to cancel their services, enabling proactive retention strategies.

## Dataset
- **Source:** Kaggle - Telco Customer Churn Dataset
- **Dataset Link:** [https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data]

## Pipeline Architecture & Methodology
1. **Data Cleaning & Preprocessing:**
   - Handled missing values and corrected data types (e.g., converting TotalCharges to numeric).
   - Dropped redundant identifiers like CustomerID.
2. **Feature Engineering & Encoding:**
   - Applied One-Hot Encoding to categorical features.
   - Performed Feature Scaling using `StandardScaler` on numerical columns.
3. **Train-Test Split:**
   - Split the data with stratification (`stratify=y`) to maintain the class distribution balance.
4. **Model Training & Evaluation:**
   - Trained multiple classification models including Logistic Regression, Random Forest, and XGBoost.
   - Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics to handle class distribution challenges.

## Results & Performance
- **Best Performing Model:** XGBoost / Random Forest
- **Key Metrics:** Achieved high predictive accuracy and reliable AUC scores, demonstrating strong generalization capabilities on unseen test data.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
