import pandas as pd
from sklearn.linear_model import LinearRegression

def weather_df():
    df = pd.read_csv("src/kumpula-weather-2017.csv")
    return df

def extract_numeric(value):
    try:
        return float(''.join(char for char in value if char.isdigit()))
    except ValueError:
        return pd.NA

def cyclists_df(station):
    # Load the dataset
    df = pd.read_csv("src/Helsingin_pyorailijamaarat.csv", sep=";")

    # Remove rows with all NaN values
    df.dropna(axis=0, how="all", inplace=True)

    # Remove columns with only missing values
    df.dropna(axis=1, how="all", inplace=True)

    # Extract year, month, and day from the "Päivämäärä" column
    date_split = df['Päivämäärä'].str.split(' ', expand=True)
    df['d'] = date_split[1].apply(extract_numeric).astype('Int64')
    df['m'] = date_split[2].map({
        'tammi': 1, 'helmi': 2, 'maalis': 3, 'huhti': 4, 'touko': 5, 'kesä': 6,
        'heinä': 7, 'elo': 8, 'syys': 9, 'loka': 10, 'marras': 11, 'joulu': 12
    }).astype('Int64')
    df['Year'] = date_split[3].apply(extract_numeric).astype('Int64')

    # Drop the original "Päivämäärä" column
    df.drop(columns=['Päivämäärä'], inplace=True)
    
    # Get the sums of cycling counts for each day
    cycling_data_daily = df.groupby(['Year', 'm', 'd'])[station].sum().reset_index()

    return cycling_data_daily

def cycling_weather(station):
    # Load DataFrames
    weather = weather_df()
    cyclists = cyclists_df(station)

    # Merge DataFrames based on the common columns "Year," "Month," and "Day"
    merged_df = pd.merge(weather, cyclists, on=['Year', 'm', 'd'], how='left')

    # Convert 'Year', 'Month', 'Day' to datetime and extract weekday
    merged_df['Weekday'] = merged_df[['Year', 'm', 'd']].astype(str).agg('-'.join, axis=1).apply(pd.to_datetime).dt.strftime('%A')

    # Drop unnecessary columns
    columns_to_drop = ['Time zone']
    merged_df.drop(columns=columns_to_drop, inplace=True)

    #rename some columns for unit test
    merged_df = merged_df.rename(columns={'m': 'Month', 'd': 'Day', 'Time': 'Hour'})

    #reorder for unit test
    """ #column_order = [
        'Year',
        'Precipitation amount (mm)',
        'Snow depth (cm)',
        'Air temperature (degC)',
        'Weekday',
        'Day',
        'Month',
        'Hour',
        'Auroransilta',
        'Eteläesplanadi',
        'Huopalahti (asema)',
        'Kaisaniemi/Eläintarhanlahti',
        'Kaivokatu',
        'Kulosaaren silta et.',
        'Kulosaaren silta po.',
        'Kuusisaarentie',
        'Käpylä',
        'Pohjoisbaana',
        'Lauttasaaren silta eteläpuoli',
        'Merikannontie',
        'Munkkiniemen silta eteläpuoli',
        'Munkkiniemi silta pohjoispuoli',
        'Heperian puisto/Ooppera',
        'Pitkäsilta itäpuoli',
        'Pitkäsilta länsipuoli',
        'Lauttasaaren silta pohjoispuoli',
        'Ratapihantie',
        'Viikintie',
        'Baana'
    ] """

    #Save the result tkumpo a CSV file
    merged_df.to_csv("output_result.csv", index=False)

    # Use forward fill to fill in missing values
    merged_df.fillna(method='ffill', inplace=True)

    return merged_df

def cycling_weather_continues(station):
    # Read weather data
    weather_data = cycling_weather(station)

    # Define the explanatory variables and response variable
    X = weather_data[['Precipitation amount (mm)', 'Snow depth (cm)', 'Air temperature (degC)']]
    y = weather_data[station]

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Get regression coefficients
    coefficients = model.coef_

    # Get the coefficient of determination (R2-score)
    r_squared = model.score(X, y)

    return coefficients, r_squared

def main():
    # Test with Baana station
    station = 'Baana'
    coefficients, r_squared = cycling_weather_continues(station)

    print(f"Measuring station: {station}")
    print(f"Regression coefficient for variable 'precipitation': {coefficients[0]:.1f}")
    print(f"Regression coefficient for variable 'snow depth': {coefficients[1]:.1f}")
    print(f"Regression coefficient for variable 'temperature': {coefficients[2]:.1f}")
    print(f"Score: {r_squared:.2f}")

if __name__ == "__main__":
    main()
