import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

    # Store the coefficients
    coefficients = model.coef_

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))

    # Print the coefficients and plot each variable against Y
    for i, (coef, column) in enumerate(zip(coefficients, X.columns), 1):
        print(f"Coefficient of {column} is {coef:.3f}")

        # Plot each variable against Y
        axs[i-1].scatter(data[column], y)
        axs[i-1].set_xlabel(column)
        axs[i-1].set_ylabel('Response Y')
        axs[i-1].set_title(f'{column} vs Response Y')

        # Add coefficient of determination label
        axs[i-1].text(0.5, 0.9, f'coeff = {coef:.3f}', horizontalalignment='center', verticalalignment='center', transform=axs[i-1].transAxes)

    plt.tight_layout()
    plt.show()

    return coefficients

def main():
    mystery_data()

if __name__ == "__main__":
    main()
