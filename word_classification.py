#!/usr/bin/env python3

from collections import Counter
import urllib.request
from lxml import etree
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

alphabet = "abcdefghijklmnopqrstuvwxyzäö-"
alphabet_set = set(alphabet)
eng_alphabet = "abcdefghijklmnopqrstuvwxyzäö-"
eng_alphabet_set = set(eng_alphabet)

# Returns a list of Finnish words
def load_finnish():
    finnish_url = "https://www.cs.helsinki.fi/u/jttoivon/dap/data/kotus-sanalista_v1/kotus-sanalista_v1.xml"
    filename = "src/kotus-sanalista_v1.xml"
    load_from_net = False
    if load_from_net:
        with urllib.request.urlopen(finnish_url) as data:
            lines = []
            for line in data:
                lines.append(line.decode('utf-8'))
        doc = "".join(lines)
    else:
        with open(filename, "rb") as data:
            doc = data.read()
    tree = etree.XML(doc)
    s_elements = tree.xpath('/kotus-sanalista/st/s')
    return list(map(lambda s: s.text, s_elements))

def load_english():
    with open("src/words", encoding="utf-8") as data:
        lines = map(lambda s: s.rstrip(), data.readlines())
    return lines

def get_features(a):
    return np.array([[word.count(letter) for letter in alphabet] for word in a])

def contains_valid_chars(s):
    return set(s).issubset(alphabet_set)

def eng_contains_valid_chars(s):
    return set(s).issubset(eng_alphabet_set)

def get_features_and_labels():
    finnish = filter(contains_valid_chars, map(lambda x: x.lower(), load_finnish()))
    english = filter(lambda x: x[0].islower(), load_english())
    english = filter(eng_contains_valid_chars, map(lambda x: x.lower(), english))
    X_finnish = get_features(finnish)
    X_english = get_features(english)
    
    Y_finnish = np.zeros(X_finnish.shape[0])
    Y_english = np.ones(X_english.shape[0])
    
    
    X = np.vstack([X_finnish, X_english])
    Y = np.hstack([Y_finnish, Y_english])
    
    return X, Y

def word_classification():
    # Get feature matrix X and target vector y
    X, y = get_features_and_labels()
    
    # Initialize Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    
    # Initialize KFold cross-validation generator
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Perform 5-fold cross-validation and get accuracy scores
    accuracy_scores = cross_val_score(clf, X, y, cv=kf)
    
    return accuracy_scores

def manual_cross_validation(n_splits=5, random_state=0):
    # Get feature matrix X and target vector y
    X, y = get_features_and_labels()
    # Initialize Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracy_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        accuracy_scores.append(accuracy)
    return accuracy_scores

def main():
    
    # Show the folds accuracy
    print("Accuracy scores are:", word_classification())

if __name__ == "__main__":
    main()
