import pandas as pd
from sklearn.linear_model import LinearRegression

def coefficient_of_determination():
    # Read the data from the file
    filename = "src/mystery_data.tsv"
    data = pd.read_csv(filename, sep='\t')

    # Extract features (first five columns) and response (last column)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Create a Linear Regression model
    model = LinearRegression()

    # Compute R2-score for all features together
    model.fit(X, y)
    r_squared_all_features = model.score(X, y)

    # Compute R2-score for each single feature
    r_squared_single_features = []
    for column in X.columns:
        model.fit(X[[column]], y)
        r_squared = model.score(X[[column]], y)
        r_squared_single_features.append(r_squared)

    return [r_squared_all_features] + r_squared_single_features

def main():
    r_squared_scores = coefficient_of_determination()
    feature_names = ['all features', 'X1', 'X2', 'X3', 'X4', 'X5']
    for feature, r_squared in zip(feature_names, r_squared_scores):
        print(f"R2-score with feature(s) {feature}: {r_squared}")

if __name__ == "__main__":
    main()
