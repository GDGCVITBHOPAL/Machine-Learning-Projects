# Titanic-Survival-Dataset-Analysis

A project depicting the analysis of the Titanic Dataset available at :
https://www.kaggle.com/c/titanic

The "Titanic-Machine Learning from Disaster" competition is an introductory kaggle competition for getting started with machine learning.

This relatively small dataset exemplifies many of the practical problems that one deals with while doing machine learning projects, namely:
1. Correlated Data
2. Missing Values
3. Different kinds of features-categorical, ordinal, numeric, alphanumeric, as well as textual.
4. Outliers

The goal here is to present and implement various methods of understanding the information contained inside the dataset that is explained with abstract information in several books and courses.


The notebook contains the following files:
1. training_data.csv : The training data which I've analyzed.
2. test_data.csv: The test data to train the classifier model on.
3. ground_truth.csv: The actual class labels for test_data.csv for confidence and accuracy score calculation.
4. titanic_survival.ipynb: The jupyter notebook which consists of explanation and code in python.


A basic outline of the workflow: 
1. Clean the dataset by removing useless and filling in the missing values.
2. Visualize individual features' correlation with the label.
3. Plot feature grids to observe biases with survival, correlation within features and wrangle accordingly.
4. Categorize the feature types.
5. Convert non numerical categorical features into numerical ones.
6. Convert quantitative features into ordinal features based on their correlation with survival.
7. Using the scaler and model object(s) of one's choice, pipeline the training process and print the average accuracy.