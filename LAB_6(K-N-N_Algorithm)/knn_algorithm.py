# -*- coding: utf-8 -*-
"""KNN_Algorithm.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hnOTBAQhNCzciUxUHqufdWORR9CCKld3

Importing the libraries
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""Load the dataset"""

# Load the dataset
dataset = pd.read_csv('KNNdataset.csv')

"""Splitting of the dataset X and Y"""

features = dataset.iloc[:,2:-1].values  # All rows and all columns except the last one (features)
labels = dataset.iloc[:,1].values     # All rows, only the last column (Species)

# Split the last row as the testing set (the one with the missing label)
features_train = features[:-1]  # Exclude the last row for training
labels_train = labels[:-1]

"""Splitting the data into training and the data set"""

X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, test_size=0.3, random_state=0)

"""Displaying the shapes of the training and test sets"""

print("Training features shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test features shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

"""Euclidean distance function"""

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

"""K-N-N prediction function"""

def knn_classifier(train_features, train_labels, test_features, k_value):
    predictions = []
    # Loop through each test point
    for test_point in test_features:
        distances = []
        # Calculate distances between the test point and all training points
        for i in range(len(train_features)):
            dist = euclidean_distance(test_point, train_features[i])
            distances.append((dist, train_labels[i]))
        # Sort based on distance and select the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = [distances[i][1] for i in range(k_value)]
        # Determine the most common class label
        most_common = Counter(k_nearest_neighbors).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

"""Function to calculate accuracy"""

def calc_accuracy(actual_labels, predicted_labels):
    return np.sum(np.array(predicted_labels) == actual_labels) / len(actual_labels)

"""Confusion matrix function"""

def create_confusion_matrix(actual_labels, predicted_labels):
    unique_classes = np.unique(actual_labels)
    confusion_mtx = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    class_to_index = {label: idx for idx, label in enumerate(unique_classes)}

    for true_label, pred_label in zip(actual_labels, predicted_labels):
        confusion_mtx[class_to_index[true_label], class_to_index[pred_label]] += 1

    return confusion_mtx

"""Predicting the labels for the entire test set"""

print("Enter k:")
k_value = int(input())
predicted_labels = knn_classifier(X_train, y_train, X_test, k_value)

"""Comparing predictions with the actual test labels"""

print("\nPredicted labels for the test set:", predicted_labels)
print("Actual labels for the test set:   ", list(y_test))

"""Calculate accuracy"""

accuracy = calc_accuracy(y_test, predicted_labels)
print(f"\nAccuracy on the test set: {accuracy * 100:.2f}%")

"""Printing Confusion Matrix"""

conf_matrix = create_confusion_matrix(y_test, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

"""Plotting K vs Accuracy and K vs Error"""

k_values_list = range(1, 21)
accuracy_list = []
error_list = []

for k_value in k_values_list:
    predicted_labels_k = knn_classifier(X_train, y_train, X_test, k_value)
    accuracy_k = calc_accuracy(y_test, predicted_labels_k)
    accuracy_list.append(accuracy_k)
    error_list.append(1 - accuracy_k)  # Error is 1 - Accuracy

"""Plotting the K vs Accuracy and K vs Error curves"""

plt.figure(figsize=(14, 6))

"""Subplot for the error"""

plt.subplot(1, 2, 2)
plt.plot(k_values_list, error_list, marker='o', linestyle='--', color='r')
plt.title('K vs Error')
plt.xlabel('K value')
plt.ylabel('Error')
plt.xticks(k_values_list)
plt.grid(True)

"""Subplot for the accuracy"""

plt.subplot(1, 2, 1)
plt.plot(k_values_list, accuracy_list, marker='o', linestyle='--', color='b')
plt.title('K vs Accuracy')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.xticks(k_values_list)
plt.grid(True)

"""Showing both the plots"""

plt.tight_layout()
plt.show()

"""Finding the optimal of K"""

optimal_k_value = k_values_list[np.argmax(accuracy_list)]
print(f"\nOptimal value of K is {optimal_k_value} with an accuracy of {max(accuracy_list) * 100:.2f}%")