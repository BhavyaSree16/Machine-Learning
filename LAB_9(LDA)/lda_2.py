# -*- coding: utf-8 -*-
"""LDA_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vaoji3IlW1SPGkZpYd0V3rr2R19CIl8f

**Import the libraries**
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

"""**Load the Dataset**"""

# Load the Wine dataset
wine = datasets.load_wine()
X = wine.data  # Features
y = wine.target  # Labels
target_names = wine.target_names  # Class names

# Display the dataset information
print("Step 1: Wine Dataset Loaded")
print("Number of Features:", X.shape[1])
print("Number of Classes:", len(target_names))
print("Feature Data (first 5 rows):\n", pd.DataFrame(X, columns=wine.feature_names).head())
print("Target Labels (first 5):\n", y[:5])
print("Class Names:", target_names)

"""**Split the Data**

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

"""**Train LDA Model**"""

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print("\nStep 3: LDA Model Trained on Training Set")

# Step 4: Evaluate the LDA Model
y_pred_lda = lda.predict(X_test)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
precision_lda = precision_score(y_test, y_pred_lda, average='weighted')
recall_lda = recall_score(y_test, y_pred_lda, average='weighted')
confusion_lda = confusion_matrix(y_test, y_pred_lda)

print("\nStep 4: LDA Model Evaluation")
print("LDA Accuracy:", accuracy_lda)
print("LDA Precision:", precision_lda)
print("LDA Recall:", recall_lda)
print("LDA Confusion Matrix:\n", confusion_lda)
print("LDA Classification Report:\n", classification_report(y_test, y_pred_lda))

"""**Evaluation of Dataset**"""

# Predict the labels on the test set
y_pred_log_reg = log_reg.predict(X_test)

# Compute the model's accuracy
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Compute the model's precision
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='weighted')

# Compute the model's recall
recall_log_reg = recall_score(y_test, y_pred_log_reg, average='weighted')

# Compute the confusion matrix
confusion_log_reg = confusion_matrix(y_test, y_pred_log_reg)

# Display the evaluation results
print("\nLogistic Regression Model Evaluation")
print("Logistic Regression Accuracy:", accuracy_log_reg)
print("Logistic Regression Precision:", precision_log_reg)
print("Logistic Regression Recall:", recall_log_reg)
print("Logistic Regression Confusion Matrix:\n", confusion_log_reg)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

"""**Comparision with Logistic Regression**"""

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000)  # Increase max_iter if convergence warning occurs
log_reg.fit(X_train, y_train)
print("\nStep 5: Logistic Regression Model Trained on Training Set")

# Evaluate the Logistic Regression model
y_pred_log_reg = log_reg.predict(X_test)

# Calculate evaluation metrics for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='weighted')
recall_log_reg = recall_score(y_test, y_pred_log_reg, average='weighted')
confusion_log_reg = confusion_matrix(y_test, y_pred_log_reg)

# Display the evaluation results for Logistic Regression
print("\nLogistic Regression Model Evaluation")
print("Logistic Regression Accuracy:", accuracy_log_reg)
print("Logistic Regression Precision:", precision_log_reg)
print("Logistic Regression Recall:", recall_log_reg)
print("Logistic Regression Confusion Matrix:\n", confusion_log_reg)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Step 6: Compare with LDA
# Assuming lda is already trained and the evaluation metrics are defined from previous steps

# Print summary of comparisons
print("\nSummary of Comparison with LDA:")
print(f"LDA Accuracy: {accuracy_lda:.2f}, Logistic Regression Accuracy: {accuracy_log_reg:.2f}")
print(f"LDA Precision: {precision_lda:.2f}, Logistic Regression Precision: {precision_log_reg:.2f}")
print(f"LDA Recall: {recall_lda:.2f}, Logistic Regression Recall: {recall_log_reg:.2f}")

# You can also compare the confusion matrices for both models
print("\nLDA Confusion Matrix:\n", confusion_lda)
print("\nLogistic Regression Confusion Matrix:\n", confusion_log_reg)