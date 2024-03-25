#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def mystery_data():
    # Read the data from the file
    data = pd.read_csv("src/mystery_data.tsv", sep='\t')

    # Extract features (first five columns) and response (last column)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Create a Linear Regression model
    model = LinearRegression(fit_intercept=False)

    # Fit the model
    model.fit(X, y)

    # Store the coeffs
    coefficients = model.coef_

    # Print the coefficients
    for i, coef in enumerate(model.coef_):
        print(f"Coefficient of X{i+1} is {coef}")

    # Determine which features are important
    threshold = 0.1
    important_features = [f"X{i+1}" for i, coef in enumerate(model.coef_) if abs(coef) > threshold]
    print("Features that might be needed to explain the response Y:", ", ".join(important_features))

    return coefficients 

def main():
    mystery_data()

if __name__ == "__main__":
    main()

