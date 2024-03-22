#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def fit_line(x, y):
    # Reshape x to a column vector
    x = x.reshape(-1, 1)
    # Create a Linear Regression model
    model = LinearRegression()
    # Fit the model
    model.fit(x, y)
    # Extract the slope and intercept from the model
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def main():
    # Example arrays
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])

    # Fit the line
    slope, intercept = fit_line(x, y)
    print("Slope:", slope)
    print("Intercept:", intercept)

    # Plot the data points and the fitted line
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, slope * x + intercept, color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title('Fitted Line using Linear Regression')
    plt.show()

if __name__ == "__main__":
    main()
