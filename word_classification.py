#!/usr/bin/env python3

from collections import Counter
import urllib.request
from lxml import etree
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

alphabet="abcdefghijklmnopqrstuvwxyzäö-"
alphabet_set = set(alphabet)

# Returns a list of Finnish words
def load_finnish():
    finnish_url="https://www.cs.helsinki.fi/u/jttoivon/dap/data/kotus-sanalista_v1/kotus-sanalista_v1.xml"
    filename="src/kotus-sanalista_v1.xml"
    load_from_net=False
    if load_from_net:
        with urllib.request.urlopen(finnish_url) as data:
            lines=[]
            for line in data:
                lines.append(line.decode('utf-8'))
        doc="".join(lines)
    else:
        with open(filename, "rb") as data:
            doc=data.read()
    tree = etree.XML(doc)
    s_elements = tree.xpath('/kotus-sanalista/st/s')
    return list(map(lambda s: s.text, s_elements))

def load_english():
    with open("src/words", encoding="utf-8") as data:
        lines=map(lambda s: s.rstrip(), data.readlines())
    return lines

def get_features(a):
    # Define the alphabet including 'ä', 'ö', and '-'
    alphabet = "abcdefghijklmnopqrstuvwxyzäö-"
    num_letters = len(alphabet)
    num_words = len(a)
    
    # Initialize feature matrix with zeros
    feature_matrix = np.zeros((num_words, num_letters), dtype=int)
    
    # Iterate over each word in the input array
    for i, word in enumerate(a):
        # Iterate over each letter in the alphabet
        for j, letter in enumerate(alphabet):
            # Count the occurrences of the letter in the word
            count = word.count(letter)
            # Assign the count to the corresponding position in the feature matrix
            feature_matrix[i, j] = count
    
    return feature_matrix

def contains_valid_chars(s):
    # Define the alphabet including 'ä', 'ö', and '-'
    alphabet = "abcdefghijklmnopqrstuvwxyzäö-"
    
    # Check if all characters in the string belong to the alphabet
    for char in s:
        if char.lower() not in alphabet:
            return False
    return True

def get_features_and_labels():
    # Load Finnish and English words
    finnish_words = load_finnish()
    english_words = load_english()
    
    # Filter Finnish words
    finnish_words = [word.lower() for word in finnish_words if contains_valid_chars(word.lower())]
    
    # Filter English words
    english_words = [word for word in english_words if word[0].islower() and contains_valid_chars(word.lower())]

    # Create labels (0 for Finnish, 1 for English)
    y = np.array([0] * len(finnish_words) + [1] * len(english_words))
    
    # Concatenate Finnish and English words
    all_words = finnish_words + english_words
    
    # Get feature matrix
    X = get_features(all_words)
    
    return X, y

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

def main():
    #testing list of 2 words
    words_list = ["apple", "banana"]

    # Convert list to numpy array
    words_array = np.array(words_list)

    # Show the folds accuracy
    print("Accuracy scores are:", word_classification())

    # Discrepancy
    my_scores = word_classification()
    test_scores = [0.86833706, 0.96897443, 0.842957,   0.87366338, 0.88320352]
    discrepancy = my_scores - test_scores
    print("Discrepancies are:", discrepancy)

if __name__ == "__main__":
    main()
