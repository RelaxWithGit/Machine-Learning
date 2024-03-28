#!/usr/bin/env python3

import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def spam_detection(random_state=0, fraction=1.0):
    # Read lines from ham.txt.gz
    with gzip.open("src/ham.txt.gz", "rt", encoding="utf-8") as ham_file:
        ham_lines = ham_file.readlines()
    
    # Read lines from spam.txt.gz
    with gzip.open("src/spam.txt.gz", "rt", encoding="utf-8") as spam_file:
        spam_lines = spam_file.readlines()
    
    # Determine number of lines to consider based on fraction
    num_ham_samples = int(len(ham_lines) * fraction)
    num_spam_samples = int(len(spam_lines) * fraction)
    
    # Take a fraction of lines from the start of each file
    ham_lines = ham_lines[:num_ham_samples]
    spam_lines = spam_lines[:num_spam_samples]
    
    # Combine ham and spam lines
    all_lines = ham_lines + spam_lines
    
    # Create labels (0 for ham, 1 for spam)
    y = [0] * num_ham_samples + [1] * num_spam_samples
    
    # Convert lines to bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_lines)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    
    # Train MultinomialNB model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Predict labels for test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate size of test sample
    test_sample_size = len(y_test)
    
    # Calculate number of misclassified samples
    misclassified_samples = sum(y_test != y_pred)
    
    return accuracy, test_sample_size, misclassified_samples

def main():
    accuracy, total, misclassified = spam_detection()
    print("Accuracy score:", accuracy)
    print("Test Sample Size:", total)
    print(f"{misclassified} messages miclassified out of {total}")

if __name__ == "__main__":
    main()
