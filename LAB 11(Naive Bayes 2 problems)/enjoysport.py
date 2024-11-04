# -*- coding: utf-8 -*-
"""Enjoysport.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1K5nN3nnUcpP5OtR3gitRzfusjzL9l-vS
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data_path = 'ENJOYSPORT.csv'  # Adjust the path to the location of your file
data = pd.read_csv(data_path)

# Encode categorical features
encoded_data = data.copy()
label_encoders = {}

# Encode each column with categorical data
for column in encoded_data.columns:
    if encoded_data[column].dtype == 'object' or encoded_data[column].dtype == 'bool':
        le = LabelEncoder()
        encoded_data[column] = le.fit_transform(encoded_data[column])
        label_encoders[column] = le  # Store label encoder for potential inverse transformation if needed

# Prepare features and target labels
X = encoded_data.drop(columns=['play'])  # Features
y = encoded_data['play']  # Target label

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naïve Bayes classifier
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=y.unique())

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Calculate error rate
error_rate = 1 - accuracy

# Print the results
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)