#!/usr/bin/env python3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def plant_classification():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Initialize Gaussian Naive Bayes classifier
    clf = GaussianNB()
    
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    
    # Predict labels for the test data
    y_pred = clf.predict(X_test)
    
    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Call the function and print the accuracy score
print("Accuracy Score:", plant_classification())

