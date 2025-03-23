from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained models
x_model = joblib.load('arabian_cyclone_model.pkl')  # Model for Arabian Sea region
y_model = joblib.load('sarima_cyclone_model.pkl')  # Model for Bay of Bengal region
z_model = joblib.load('land_cyclone_model.pkl')  # Model for Land Region

# Define state-to-region mapping
state_to_region = {
    # Arabian Sea States
    'Gujarat': 'Arabian Sea',
    'Maharashtra': 'Arabian Sea',
    'Goa': 'Arabian Sea',
    'Karnataka': 'Arabian Sea',
    'Kerala': 'Arabian Sea',
    # Arabian Sea Union Territories
    'Daman and Diu': 'Arabian Sea',
    'Lakshadweep': 'Arabian Sea',
    # Bay of Bengal States
    'Andhra Pradesh': 'Bay of Bengal',
    'Odisha': 'Bay of Bengal',
    'West Bengal': 'Bay of Bengal',
    'Tamil Nadu': 'Bay of Bengal',
    # Bay of Bengal Union Territories
    'Puducherry': 'Bay of Bengal',
    'Andaman and Nicobar Islands': 'Bay of Bengal',
    # Land Region (all other states and union territories)
    # Add more states/UTs as needed
}

# Pydantic model for request input
class CityInput(BaseModel):
    city: str

# Function to get state from city
def get_state_from_city(city):
    geolocator = Nominatim(user_agent="city_classifier")
    location = geolocator.geocode(city + ", India")
    if location:
        address = location.raw.get('address', {})
        state = address.get('state', '')
        if not state:
            # If state is not found, try to extract it from the display name
            display_name = location.raw.get('display_name', '')
            if ', Maharashtra,' in display_name:
                state = 'Maharashtra'
            elif ', Gujarat,' in display_name:
                state = 'Gujarat'
            elif ', Goa,' in display_name:
                state = 'Goa'
            elif ', Karnataka,' in display_name:
                state = 'Karnataka'
            elif ', Kerala,' in display_name:
                state = 'Kerala'
            elif ', Daman and Diu,' in display_name:
                state = 'Daman and Diu'
            elif ', Lakshadweep,' in display_name:
                state = 'Lakshadweep'
            elif ', Andhra Pradesh,' in display_name:
                state = 'Andhra Pradesh'
            elif ', Odisha,' in display_name:
                state = 'Odisha'
            elif ', West Bengal,' in display_name:
                state = 'West Bengal'
            elif ', Tamil Nadu,' in display_name:
                state = 'Tamil Nadu'
            elif ', Puducherry,' in display_name:
                state = 'Puducherry'
            elif ', Andaman and Nicobar Islands,' in display_name:
                state = 'Andaman and Nicobar Islands'
        return state
    else:
        raise ValueError(f"Could not find state for city: {city}")

# Function to classify region based on state
def classify_region(city):
    state = get_state_from_city(city)
    # Extract the state name (e.g., "Maharashtra" from "Mumbai Suburban, Maharashtra, India")
    state_name = state.split(',')[0].strip() if state else ''
    return state_to_region.get(state_name, 'Land Region')  # Default to Land Region if state not found

# Function to select model and dataset based on city
def select_model_and_dataset(city):
    region = classify_region(city)
    if region == 'Arabian Sea':
        return x_model, 'arabiancyclones.csv'
    elif region == 'Bay of Bengal':
        return y_model, 'bobcyclones.csv'
    else:
        return z_model, 'landcyclones.csv'

# Function to generate forecasts and return results
def generate_forecast(city):
    # Select model and dataset based on city
    model, dataset_file = select_model_and_dataset(city)

    # Load the dataset for the region
    data = pd.read_csv(dataset_file)

    # Remove rows where 'Year' is not a valid year (e.g., "Total ")
    data = data[data['Year'].astype(str).str.isdigit()]

    # Convert Year to datetime and set as index
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')

    # Melt the data to create a seasonal time series
    seasonal_data = data.melt(id_vars=['Year'], value_vars=['JF', 'MAM', 'JJAS', 'OND'],
                              var_name='Season', value_name='Cyclone_Count')

    # Map seasons to months for proper datetime indexing
    season_to_month = {
        'JF': 1,   # January-February (use January as the start month)
        'MAM': 4,  # March-May (use March as the start month)
        'JJAS': 6, # June-September (use June as the start month)
        'OND': 10  # October-December (use October as the start month)
    }
    seasonal_data['Month'] = seasonal_data['Season'].map(season_to_month)

    # Create a datetime index for the seasonal data
    seasonal_data['Date'] = pd.to_datetime(seasonal_data['Year'].dt.year.astype(str) + '-' + seasonal_data['Month'].astype(str) + '-01')
    seasonal_data.set_index('Date', inplace=True)

    # Sort the data by date
    seasonal_data = seasonal_data.sort_index()

    # Explicitly set the frequency of the time series index
    seasonal_data = seasonal_data.asfreq('QS-JAN')  # Quarterly frequency starting in January

    # Use the 'Cyclone_Count' column as the seasonal time series
    time_series = seasonal_data['Cyclone_Count']

    # Forecast the next 8 seasons (2 years starting from 2025)
    forecast_steps = 8
    forecast = model.forecast(steps=forecast_steps)

    # Round forecasted values to whole numbers
    forecast = np.round(forecast).astype(int)

    # Create a DataFrame for the forecast
    forecast_years = [2025, 2025, 2025, 2025, 2026, 2026, 2026, 2026]  # Years for the forecast
    forecast_seasons = ['JF', 'MAM', 'JJAS', 'OND', 'JF', 'MAM', 'JJAS', 'OND']  # Seasons for the forecast

    forecast_df = pd.DataFrame({
        'Year': forecast_years,
        'Season': forecast_seasons,
        'Forecasted_Cyclone_Count': forecast
    })

    # Calculate cyclone probabilities
    max_cyclone_count = time_series.max()
    forecast_df['Cyclone_Probability'] = np.round((forecast_df['Forecasted_Cyclone_Count'] / max_cyclone_count) * 100).astype(int)

    # Convert forecast DataFrame to JSON
    forecast_json = forecast_df.to_dict(orient='records')

    return {
        "city": city,
        "region": classify_region(city),
        "forecast": forecast_json
    }

# FastAPI endpoint
@app.post("/forecast")
def forecast(city_input: CityInput):
    try:
        result = generate_forecast(city_input.city)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)