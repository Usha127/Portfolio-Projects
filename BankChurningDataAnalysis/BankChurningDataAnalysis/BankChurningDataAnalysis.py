# This project reads a CSV file containing bank customer data, processes it, and builds a logistic regression model to predict customer churn.
# It includes data cleaning, feature encoding, scaling, and model evaluation.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#Read the CSV file
df = pd.read_csv('../Data/Bank_churn_modelling.csv')
print(df.head())
print('Shape of the df:', df.shape)
print('Description of the df:\n', df.describe())
print("----------------------------------------------------------- ")

# Data Cleaning Process
# Drop columns that are not needed for analysis
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
print(df.head())
print("----------------------------------------------------------- ")

# Check for null values
print('Null values in the df:\n', df.isnull().sum())
print("----------------------------------------------------------- ")

# Check for duplicates
print('Duplicates in the df:', df.duplicated().sum())
print("----------------------------------------------------------- ")

# Remove duplicates if any
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print('Duplicates removed. New shape of the df:', df.shape)
print("----------------------------------------------------------- ")

# Print column names and data types
print('Column names and data types:\n', df.dtypes)
print("----------------------------------------------------------- ")

#print first 5 rows of the dataset with all columns
print('First 5 rows of the dataset:\n', df.head(5))
print("----------------------------------------------------------- ")

''' One way of categorising columns in to numerical and non-numerical columns
# Create graphs for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('Numerical columns in the df:', numerical_cols)
print("----------------------------------------------------------- ")
# Create list of categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print('Categorical columns in the df:', categorical_cols)
print("----------------------------------------------------------- ")
'''
# Another way of categorising columns in to discrete and continuous columns
# Create discrete column list
discrete_cols = ['Geography','Gender','HasCrCard','IsActiveMember','Exited']
print('Discrete columns in the df:', discrete_cols)
print("----------------------------------------------------------- ")

# Create continuous column list
continuous_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
print('Continuous columns in the df:', continuous_cols)
print("----------------------------------------------------------- ")




# Function to create histograms for continuous columns
def create_histograms(df, continuous_cols):
    for col in continuous_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        # Create a directory to save images if it doesn't exist
        if not os.path.exists('../Images'):
            os.makedirs('../Images')
        plt.savefig(f'../Images/Histogram_Distribution_{col}.jpg')
        plt.show()
    return

# Function to create count plots for discrete columns
def create_count_plots(df, discrete_cols):
    for col in discrete_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col)
        plt.title(f'Count plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.grid(True)
        # Create a directory to save images if it doesn't exist
        if not os.path.exists('../Images'):
            os.makedirs('../Images')
        plt.savefig(f'../Images/Count_Plot_{col}.jpg')
        plt.show()
    return


# Function to compare the distribution of 'Exited' column with discrete columns using Bar plots y axis as count of the column
def compare_exited_with_discrete_columns(df, discrete_cols):
    for col in discrete_cols:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x=col, hue='Exited', palette='viridis')
        plt.title(f'Count plot of {col} vs Exited')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.grid(True)
        # Create a directory to save images if it doesn't exist
        if not os.path.exists('../Images'):
            os.makedirs('../Images')
        plt.savefig(f'../Images/Count_Plot_{col}_vs_Exited.jpg')
        plt.show()
    return

# Create a chart to compare the 'Exited' column vs 'CreditScore' column
def compare_exited_vs_credit_score(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='CreditScore', y='Exited')
    plt.title(f'Bar plot of CreditScore vs Exited')
    plt.xlabel('CreditScore')
    plt.ylabel('Exited')
    plt.grid(True)
    # Create a directory to save images if it doesn't exist
    if not os.path.exists('../Images'):
        os.makedirs('../Images')
    plt.savefig(f'../Images/BarPlot_Exited_vs_CreditScore.jpg')
    plt.show()
    return

# Create a chart to compare the 'Exited' vs 'CreditScore' column using a scatter plot
def compare_exited_vs_credit_score_scatter(df):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='CreditScore', y='Exited')
    plt.title(f'Scatter plot of CreditScore vs Exited')
    plt.xlabel('CreditScore')
    plt.ylabel('Exited')
    plt.grid(True)
    # Create a directory to save images if it doesn't exist
    if not os.path.exists('../Images'):
        os.makedirs('../Images')
    plt.savefig(f'../Images/ScatterPlot_Exited_vs_CreditScore.jpg')
    plt.show()
    return

# Create a histogram for each continuous column vs 'Exited' column for both 'Exited'=0 and 'Exited'=1
def create_histograms_vs_exited_0_and_1(df, continuous_cols):
    for col in continuous_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=col, hue='Exited', multiple='stack', kde=True)
        plt.title(f'Distribution of {col} by Exited')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        # Create a directory to save images if it doesn't exist
        if not os.path.exists('../Images'):
            os.makedirs('../Images')
        plt.savefig(f'../Images/Histogram_Distribution_{col}_by_Exited_0_and_1.jpg')
        plt.show()
    return


print("----------------------------------------------------------- ")
# Call method to create histograms for continuous columns
create_histograms(df, continuous_cols)
# Call method to create count plots for discrete columns
create_count_plots(df, discrete_cols)
# Call method to compare the distribution of 'Exited' column with discrete columns
compare_exited_with_discrete_columns(df, discrete_cols)
# Call method to compare the 'Exited' column vs 'CreditScore' column
compare_exited_vs_credit_score(df)
# Call method to compare the 'Exited' vs 'CreditScore' column using a scatter plot
compare_exited_vs_credit_score_scatter(df)
# Call method to create histograms for each continuous column vs 'Exited' column for both 'Exited'=0 and 'Exited'=1
create_histograms_vs_exited_0_and_1(df, continuous_cols)
print("----------------------------------------------------------- ")
# Data Preprocessing
# Data Preprocessing after EDA and Data Cleaning

# Drop columns hascrcard, tenure, and estimatedsalary from df
df = df.drop(columns=['HasCrCard', 'Tenure', 'EstimatedSalary'])
print('DataFrame after dropping columns:\n', df.head())
print("----------------------------------------------------------- ")

x = df.drop('Exited', axis=1)
y = df['Exited']

print('x.head():\n', x.head())
print("----------------------------------------------------------- ")
print('y.head():\n', y.head())  
print("----------------------------------------------------------- ")

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#The OneHotEncoder from sklearn.preprocessing transforms categorical features 
# into a one-hot numeric array. It converts each category within a feature 
# into a binary representation, creating a new column for each unique category. 
# The input should be an array-like object containing integers or strings. 
# This is useful for preparing categorical data for machine learning algorithms 
# that require numerical input. This encoding scheme allows the model to 
# treat each category independently, which is crucial for accurate model training.
# The StandardScaler from sklearn.preprocessing standardizes features by removing the mean 
# and scaling to unit variance. It transforms the data such that each feature
# has a mean of 0 and a standard deviation of 1, which is important for algorithms
# that are sensitive to the scale of the data.
encoder=ColumnTransformer([('one',OneHotEncoder(),[1,2]),('sc',StandardScaler(),[0,3,4,5])],remainder='passthrough')
newdata=encoder.fit_transform(x)
print(type(newdata))
print("----------------------------------------------------------- ")
print('newdata:\n', newdata)    
print("----------------------------------------------------------- ")
# Convert the newdata to a DataFrame with appropriate column names
#column_names = encoder.get_feature_names_out()
#df1 = pd.DataFrame(newdata, columns=column_names)
df1 = pd.DataFrame(newdata)
print('DataFrame after encoding and scaling:\n', df1.head())
print("----------------------------------------------------------- ")

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df1, y, test_size=0.2)
print('Shape of x_train:', x_train.shape)
print('Shape of x_test:', x_test.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_test:', y_test.shape)
print("----------------------------------------------------------- ")
# Model Building - Logistic Regression
from sklearn.linear_model import LogisticRegression
# Create a Logistic Regression model
model = LogisticRegression()
# Fit the model on the training data
model.fit(x_train, y_train)
print('Model trained successfully.')
print("----------------------------------------------------------- ")

# Predict the 'Exited' on the test set
y_pred = model.predict(x_test)
print('Predictions on the test set:', y_pred)
print("----------------------------------------------------------- ")

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model:', accuracy)
print("----------------------------------------------------------- ")
print('Classification report:\n', classification_report(y_test, y_pred))
print("----------------------------------------------------------- ")
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))  
print("----------------------------------------------------------- ")

# Save the model using joblib
import joblib
joblib.dump(model, 'bank_churn_model.pkl')
print('Model saved as bank_churn_model.pkl')
print("----------------------------------------------------------- ")
