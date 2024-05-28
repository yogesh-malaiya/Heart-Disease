# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# Get started with interactive Python!
# Supports Python Modules: builtins, math,pandas, scipy 
# matplotlib.pyplot, numpy, operator, processing, pygal, random, 
# re, string, time, turtle, urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Load the dataset
data = pd.read_csv('heartdisease.csv')

# Data Overview
print(data.head())
print(data.info())
print(data.describe())

# Data Filtering (if necessary, e.g., removing null values)
data = data.dropna()

# Exploratory Data Analysis (EDA)
# Distribution of the target variable
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=data)
plt.title('Distribution of Target Variable')
plt.savefig('target_distribution.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Pairplot of the features
sns.pairplot(data, hue='target', vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
plt.savefig('pairplot.png')
plt.show()

# Data Visualization
# Age distribution by target
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='target', multiple='stack', kde=True)
plt.title('Age Distribution by Target')
plt.savefig('age_distribution.png')
plt.show()

# Train-Test Split
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

