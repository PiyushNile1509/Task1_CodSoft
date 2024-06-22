# DS_Task1_CodSoft
**Title: Titanic Survival Prediction with Machine Learning**

*Description:*

This repository implements machine learning algorithms to predict passenger survival on the infamous Titanic voyage. It leverages the well-known Titanic dataset from Kaggle, which contains information about passengers such as age, sex, class, and their fate during the disaster.

*Objective:*

The primary goal is to build a model that can accurately classify whether a passenger would have survived the Titanic sinking based on the available historical data. This project serves as a practical introduction to machine learning techniques and data analysis in a historical context.

*Libraries Used:*

The following important libraries were used for this project:
-numpy
-pandas
-matplotlib.pyplot
-seaborn
-sklearn.preprocessing.LabelEncoder
-sklearn.model_selection.train_test_split
-sklearn.linear_model.LogisticRegression

*Data Exploration and Preprocessing*

1.The dataset was loaded using pandas as a DataFrame, and its shape and a glimpse of the first 10 rows were displayed using df.shape and df.head(10).
2.Descriptive statistics for the numerical columns were displayed using df.describe() to get an overview of the data, including missing values.
3.The count of passengers who survived and those who did not was visualized using sns.countplot(x=df['Survived']).
4.The count of survivals was visualized with respect to the Pclass using sns.countplot(x=df['Survived'], hue=df['Pclass']).
5.The count of survivals was visualized with respect to the gender using sns.countplot(x=df['Sex'], hue=df['Survived']).
6.The survival rate by gender was calculated and displayed using df.groupby('Sex')[['Survived']].mean().
7.The 'Sex' column was converted from categorical to numerical values using LabelEncoder from sklearn.preprocessing.
8.After encoding the 'Sex' column, non-required columns like 'Age' were dropped from the DataFrame.

