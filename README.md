# Predicting the Diagnosis of Internet Gaming Disorder using Supervised Machine Learning

# Overview
This GitHub repository hosts the code for my data science thesis project, which aims to classify Internet Gaming Disorder (IGD) in gamers. The project investigates the efficacy of various machine learning models to address this binary classification problem, providing insights into the complexities and nuances of predictive modeling in the context of behavioral disorders.

# 1. Research Objectives
The main objective of this research is to determine the most effective machine learning model for diagnosing IGD among adult gamers. I evaluate both simple and complex models, including Logistic Regression, Random Forest, XGBoost, and AdaBoost, to establish which model achieves the best predictive performance.

# 2. Methodology

### Algorithms Selection
Chosen Logistic Regression and Random Forest for their proven past performance and simple complexity, and XGBoost and AdaBoost for their robustness in handling non-linear relationships and iterative optimization.

### Data Preparation
Utilized a dataset from The Cairnmillar Institute comprising demographic and behavioral data, as well as psychometric evaluations from 1,032 gamers.

### Feature Selection
Conducted in three iterations, expanding the feature set from demographic and behavioral data to include psychometric scores.

### Model Training and Hyperparameter Tuning
Applied 5-fold cross-validation using Grid Search for Logistic Regression and Random Search for the ensemble models.

### Evaluation
Focused on F1-score and precision-recall curves to balance the trade-off between Precision and Recall, particularly important due to class imbalance in the dataset.

# 3. Tools and Technologies

### **Programming Language**: Python 3.9.12
### **Key Libraries**: matplotlib, imblearn, xgboost, scikit-learn

#### Data Manipulation and Analysis
- **Pandas**: Provides data structures and data analysis tools.
- **NumPy**: Supports large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

#### Machine Learning and Data Preprocessing
- **Scikit-learn**: Offers tools for data mining and data analysis, including pre-processing, cross-validation, and training algorithms.
- **Imbalanced-learn (Imblearn)**: Provides techniques for dealing with imbalanced datasets.

#### Plotting and Visualization
- **Matplotlib**: A plotting library for creating static, interactive, and animated visualizations in Python.
- **Seaborn**: Based on matplotlib, provides a high-level interface for drawing attractive and informative statistical graphics.

#### Ensemble Methods and Classification Algorithms
- **XGBoost**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.
- **SciPy**: Used for scientific and technical computing.

#### File I/O
- **Pyreadstat**: A Python package to read and write SAS, SPSS, and Stata files into/from pandas data frames.

#### Python Standard Library
- **Collections**: Offers specialized container datatypes providing alternatives to Python’s general purpose built-in containers. 

# 4. Dataset

Data is provided by The Cairnmillar Institute and includes extensive behavioral and psychometric attributes. Features include gender, age, gaming hours, and scores from various psychological assessments like the IGD9-SF scale. Available at https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-x3e-4452


# 5. Results

This section synthesizes the findings from various stages of model evaluation and hyperparameter tuning, providing a clear overview of which models performed best and how different feature sets impacted the predictive performance.

## 5.1. Best Hyperparameters

The optimal hyperparameters for each model, selected based on achieving the highest F1-scores, are detailed as follows:

- **Logistic Regression**: Best achieved with `C=0.15` and the `saga` solver.
- **Random Forest**: Optimal parameters include `entropy` as the impurity measure, `max_depth=11`, `max_features=30`, and `n_estimators=92`.
- **XGBoost**: Configured with `colsample_bytree=0.161`, `gamma=0.068`, `learning_rate=0.565`, `max_depth=4`, `min_child_weight=3`, `reg_alpha=0.1`, and `n_estimators=173`.
- **AdaBoost**: Found most effective with a `learning_rate of 0.671` and `n_estimators=289`.

## 5.2. Determining the Best Prediction Model

The comprehensive evaluation of F1-scores across all feature sets showed that:
- **Logistic Regression** emerged as the leading model in Feature Set 3, with the highest F1-score of `0.471`, indicative of its superior capability in handling a diverse range of features effectively.
- Among ensemble methods, **XGBoost** followed closely with an F1-score of `0.452`, showcasing its strength in feature handling and prediction accuracy.

## 5.3. Feature Importance

The analysis of feature importance in the XGBoost model highlighted several key predictors:
- **Most Influential**: 'Education', 'Genre (Music/Dance)', and 'Anxiety Score' were among the top contributors to the model's predictive ability.
- **Least Influential**: Features like 'Country' and various 'Occupation' classes showed minimal impact on the model's performance.

## Summary

This research identified **Logistic Regression** and **XGBoost** as the most effective models for predicting Internet Gaming Disorder, with their performance significantly enhanced by the inclusion of psychometric features. The detailed analysis of feature importance provided valuable insights into the predictors that are most critical in diagnosing IGD effectively.
