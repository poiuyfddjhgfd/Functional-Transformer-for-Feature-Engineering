# Functional Transformer for Feature Engineering

This project demonstrates how to use `FunctionTransformer` from scikit-learn to apply logarithmic transformations to numerical features, improving machine learning model performance.

## 📋 Project Overview

The project analyzes the Titanic dataset, focusing on the 'Age' and 'Fare' features to predict survival. It compares model performance before and after applying logarithmic transformations using scikit-learn's `FunctionTransformer`.

## 🎯 Key Features

- **Data Preprocessing**: Handles missing values in the 'Age' column
- **Feature Transformation**: Applies `np.log1p` transformation using `FunctionTransformer`
- **Model Comparison**: Evaluates Logistic Regression and Decision Tree classifiers
- **Visualization**: Includes distribution plots and QQ plots to assess normality
- **Cross-Validation**: Uses 10-fold cross-validation for robust performance evaluation

## 📊 Dataset

The project uses the Titanic dataset with the following features:
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Age**: Passenger age (with missing values filled using mean imputation)
- **Fare**: Ticket fare

## 🛠️ Installation & Dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
