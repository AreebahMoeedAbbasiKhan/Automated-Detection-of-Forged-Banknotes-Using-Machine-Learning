# Automated Detection of Forged Banknotes Using Machine Learning

## Project Overview

This project explores how machine learning can be used to automate the detection of forged banknotes. Using statistical image features extracted from banknotes, we apply data analysis and clustering techniques to identify patterns that separate genuine and counterfeit notes.

The objective is to evaluate whether an automated system can assist banks in reducing fraud and improving operational efficiency.

---

## Dataset

The dataset contains numerical features extracted from scanned banknote images:

* Variance
* Skewness
* Kurtosis
* Entropy
* Class (Genuine or Forged)

These features are derived from image processing techniques and represent measurable statistical characteristics of the banknotes.

---

## Project Workflow

1. Data Loading and Cleaning
2. Column Standardization and Error Handling
3. Exploratory Data Analysis (EDA)
4. Feature Scaling using StandardScaler
5. K-Means Clustering (2 clusters)
6. Stability Testing with Multiple Initializations
7. Visualization of Clustering Results

---

## Methods Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* K-Means Clustering
* Data Standardization

---

## Results

The clustering analysis revealed that the dataset naturally separates into two distinct groups. Visualizations show clear cluster separation, and repeated runs demonstrate stability in clustering results.

These findings suggest that automated detection of forged banknotes using machine learning is feasible.

---

## Key Features

* Robust data cleaning pipeline
* Automatic column correction
* Multiple clustering runs for stability analysis
* Visual comparison of clustering outcomes
* Ready for extension into supervised learning models

---

## Future Improvements

* Implement supervised classification models (Logistic Regression, Decision Trees)
* Add performance metrics (accuracy, precision, recall)
* Deploy as a real-time fraud detection system

---

## Author
Areebah Abbasi
Data Science Project

