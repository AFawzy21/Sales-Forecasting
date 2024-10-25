import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import GRU
from sklearn.metrics import mean_absolute_error as mae
from flask import Flask, request, jsonify
import re
import json


app = Flask(__name__)


def ProphetMethod(df,forecast_from,forecast_to,unique_items):
    forecasts = []
    # for item_id in unique_items:
        # Filter data for the specific item
        # item_data = df[df['ItemID'] == item_id]
    # Initialize and fit the model
    model = Prophet(seasonality_mode='additive')
    model.fit(df[['ds', 'y']])
    # Create a dataframe for future dates
    future_date = pd.date_range(start=forecast_from, end=forecast_to, freq='M')
    future = pd.DataFrame({'ds': future_date})
    # Predict future sales
    forecast = model.predict(future)
    forecasts = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    print(forecasts)
    return forecasts

def SARIMAXMethod(df,forecast_from,forecast_to,unique_items):
    forecasts = []
    # for item_id in unique_items:
        # Filter data for the specific item
        # item_data = df[df['ItemID'] == item_id]
    item_data = df.copy()
    item_data.set_index('ds', inplace=True)
    # Fit the ARIMA model
    model = SARIMAX(item_data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    # Create a dataframe for future dates
    future_date = pd.date_range(start=forecast_from, end=forecast_to, freq='M')
    future = pd.DataFrame({'ds': future_date})
    future.set_index('ds', inplace=True)
    # Predict future sales
    forecast = model_fit.get_forecast(steps=len(future_date))
    forecast_df = forecast.conf_int()
    forecast_df['yhat'] = forecast.predicted_mean
    forecast_df['yhat_lower'] = forecast_df['lower y']
    forecast_df['yhat_upper'] = forecast_df['upper y']
    forecast_df = forecast_df.drop(['lower y', 'upper y'], axis=1)
    forecast_df = forecast_df.reset_index()
    
    forecasts = forecast_df.to_dict(orient='records')
    print(forecasts)
    return forecasts

def ARIMAMethod(df,forecast_from,forecast_to,unique_items):
    forecasts = []

    # for item_id in unique_items:
        # Filter data for the specific item
        # item_data = df[df['ItemID'] == item_id]
    item_data = df.copy()
    item_data.set_index('ds', inplace=True)
    # Fit the ARIMA model
    model = ARIMA(item_data['y'], order=(1, 1, 1))
    model_fit = model.fit()
    # Create a dataframe for future dates
    future_date = pd.date_range(start=forecast_from, end=forecast_to, freq='M')
    future = pd.DataFrame({'ds': future_date})
    future.set_index('ds', inplace=True)
    # Predict future sales
    forecast = model_fit.get_forecast(steps=len(future_date))
    forecast_df2 = forecast.conf_int()
    forecast_df2['yhat'] = forecast.predicted_mean
    forecast_df2['yhat_lower'] = forecast_df2['lower y']
    forecast_df2['yhat_upper'] = forecast_df2['upper y']
    forecast_df2 = forecast_df2.drop(['lower y', 'upper y'], axis=1)
    forecast_df2 = forecast_df2.reset_index()
    
    forecasts = forecast_df2.to_dict(orient='records')
    print(forecasts)
    return forecasts

def RandomForestMethod(df,forecast_from,forecast_to,unique_items):
    forecasts = []

    # for item_id in unique_items:
        # Filter data for the specific item
        # item_data = df[df['ItemID'] == item_id]
    item_data = df.copy()
    item_data.set_index('ds', inplace=True)
    # Prepare the data for RandomForestRegressor
    item_data['ds'] = item_data.index
    item_data['ds'] = item_data['ds'].map(pd.Timestamp.toordinal)
    X = item_data[['ds']]
    y = item_data['y']
    # Fit the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X, y)
    # Create a dataframe for future dates
    future_date = pd.date_range(start=forecast_from, end=forecast_to, freq='M')
    future = pd.DataFrame({'ds': future_date})
    future.set_index('ds', inplace=True)
    future['ds'] = future.index
    future['ds'] = future['ds'].map(pd.Timestamp.toordinal)
    X_future = future[['ds']]
    # Predict future sales
    forecast_df3 = pd.DataFrame(model.predict(X_future), columns=['yhat'])
    forecast_df3['ds'] = future_date
    forecast_df3 = forecast_df3.reset_index()
    forecasts = forecast_df3.to_dict(orient='records')

    print(forecasts)
    return forecasts

def XGBoostMethod(df,forecast_from,forecast_to,unique_items):
    forecasts = []

    # for item_id in unique_items:
        # Filter data for the specific item
        # item_data = df[df['ItemID'] == item_id]
    item_data = df.copy()
    item_data.set_index('ds', inplace=True)
    # Prepare the data for XGBRegressor
    item_data['ds'] = item_data.index
    item_data['ds'] = item_data['ds'].map(pd.Timestamp.toordinal)
    X = item_data[['ds']]
    y = item_data['y']
    # Fit the XGBRegressor model
    model = XGBRegressor(n_estimators=1000, random_state=42)
    model.fit(X, y)
    # Create a dataframe for future dates
    future_date = pd.date_range(start=forecast_from, end=forecast_to, freq='M')
    future = pd.DataFrame({'ds': future_date})
    future.set_index('ds', inplace=True)
    future['ds'] = future.index
    future['ds'] = future['ds'].map(pd.Timestamp.toordinal)
    X_future = future[['ds']]
    # Predict future sales
    xgb_forecast_df = pd.DataFrame(model.predict(X_future), columns=['yhat'])
    xgb_forecast_df['ds'] = future_date
    xgb_forecast_df = xgb_forecast_df.reset_index()
    forecasts = xgb_forecast_df.to_dict(orient='records')

    print(forecasts)
    return forecasts

def LSTMMethod(df,forecast_from,forecast_to,unique_items,adjusted):
    forecasts = []

    # for item_id in unique_items:
        # Filter data for the specific item
        # item_data = df[df['ItemID'] == item_id]
    item_data = df.copy()
    item_data.set_index('ds', inplace=True)
    # Prepare the data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    item_data_scaled = scaler.fit_transform(item_data[['y']])
    # Create a function to prepare the dataset for LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    look_back = 12
    X, y = create_dataset(item_data_scaled, look_back)
    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=10, batch_size=1, verbose=2)
    # Create a dataframe for future dates
    future_date = pd.date_range(start=forecast_from, end=forecast_to, freq='M')
    future = pd.DataFrame({'ds': future_date})
    future.set_index('ds', inplace=True)
    # Prepare the future data for prediction
    last_values = item_data_scaled[-look_back:]
    future_predictions = []
    for _ in range(len(future_date)):
        prediction = model.predict(np.reshape(last_values, (1, look_back, 1)))
        future_predictions.append(prediction[0][0])
        last_values = np.append(last_values[1:], prediction)[-look_back:].reshape(-1, 1)
    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    # Create a dataframe for the predictions
    lstm_forecast_df = pd.DataFrame(future_predictions, columns=['yhat'])
    lstm_forecast_df['ds'] = future_date
    lstm_forecast_df = lstm_forecast_df.reset_index()
    lstm_forecast_df['new_yhat'] = lstm_forecast_df['yhat'] * adjusted / 100
    forecasts = lstm_forecast_df.to_dict(orient='records')



    print(forecasts)
    return forecasts

def forecast_dict_to_df(forecast_dict, method_name, include_new_yhat=False):
    df_list = []
    for forecasts in forecast_dict:
        df = pd.DataFrame([forecasts])
        
        # df['ItemID'] = item_id
        if include_new_yhat:
            df = df.rename(columns={'yhat': method_name, 'new_yhat': 'LSTM_Adjusted'})
        else:
            df = df.rename(columns={'yhat': method_name})
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

@app.route('/getForecasting', methods=['POST'])
def getForecasting():
    try:
        raw_data = request.data
        # print({"Raw request data": raw_data})  # Print raw data for debugging
        # Decode the raw data to a string
        raw_data_str = raw_data.decode('utf-8')

        # Remove non-breaking spaces or other problematic characters
        clean_data_str = re.sub(r'[\u00A0]', ' ', raw_data_str)  # Replacing NBSP with a regular space
        # Attempt to parse the cleaned JSON string
        data = json.loads(clean_data_str)
        forecast_from = data['forecast_from']
        forecast_to = data['forecast_to']
        itemdata=data['data']
        adjusted=data['adjusted_percent']

        # Convert input data to DataFrame
        df = pd.DataFrame(itemdata)
        clmns = df.drop(columns=['Value']).columns.tolist()
        # Group by Year, Month, and ItemID and sum the values
        df = df.groupby(clmns).sum().reset_index()

        # Combine year and month into a single date column
        df['ds'] = pd.to_datetime(df[clmns].assign(month=1,day=1))

        # Rename the sales column to 'y' as required by Prophet
        df = df.rename(columns={'Value': 'y'})
        df['y'] = pd.to_numeric(df['y'], errors='coerce')

        # Drop rows with missing values in 'y'
        df = df.dropna(subset=['y'])
        # Get unique item IDs
        # unique_items = df['ItemID'].unique()
        unique_items = None
        # print(df)
        print("//////////////////")
        print("Prophet")
        forecast_methodProphet = ProphetMethod(df,forecast_from,forecast_to,unique_items)
        print("//////////////////")
        print("SARIMAX")
        forecast_methodSARIMAX = SARIMAXMethod(df,forecast_from,forecast_to,unique_items)
        print("//////////////////")
        print("ARIMA")
        forecast_methodARIMA = ARIMAMethod(df,forecast_from,forecast_to,unique_items)
        print("//////////////////")
        print("Random Forest")
        forecast_methodRandomForest = RandomForestMethod(df,forecast_from,forecast_to,unique_items)
        print("//////////////////")
        print("XGBoost")
        forecast_methodXGBoost = XGBoostMethod(df,forecast_from,forecast_to,unique_items)
        print("//////////////////")
        print("LSTM")
        forecast_methodLSTM = LSTMMethod(df,forecast_from,forecast_to,unique_items,adjusted)
        print("//////////////////")
        
        # Convert each forecast dictionary to DataFrame
        df_prophet = forecast_dict_to_df(forecast_methodProphet, 'Prophet')
        df_sarimax = forecast_dict_to_df(forecast_methodSARIMAX, 'SARIMAX')
        df_arima = forecast_dict_to_df(forecast_methodARIMA, 'ARIMA')
        df_rf = forecast_dict_to_df(forecast_methodRandomForest, 'Random Forest')
        df_xgb = forecast_dict_to_df(forecast_methodXGBoost, 'XGBoost')
        df_lstm = forecast_dict_to_df(forecast_methodLSTM, 'LSTM', include_new_yhat=True)

        df_sarimax = df_sarimax.reset_index().rename(columns={'index': 'ds'})
        df_arima = df_arima.reset_index().rename(columns={'index': 'ds'})

        for dff in [df_prophet, df_sarimax, df_arima, df_rf, df_xgb, df_lstm]:
            dff['Year'] = dff['ds'].dt.year
            dff['Month'] = dff['ds'].dt.month
        # Merge all DataFrames on 'Year', 'Month', and 'ItemID' with suffixes to avoid column name conflicts
        df_combined = df_prophet.merge(df_sarimax, on=['Year', 'Month'], how='outer', suffixes=('', '_sarimax'))
        df_combined = df_combined.merge(df_arima, on=['Year', 'Month'], how='outer', suffixes=('', '_arima'))
        df_combined = df_combined.merge(df_rf, on=['Year', 'Month'], how='outer', suffixes=('', '_rf'))
        df_combined = df_combined.merge(df_xgb, on=['Year', 'Month'], how='outer', suffixes=('', '_xgb'))
        df_combined = df_combined.merge(df_lstm, on=['Year', 'Month'], how='outer', suffixes=('', '_lstm'))

        # Extract Year and Month from 'ds'
        df_combined['Year'] = df_combined['ds'].dt.year
        df_combined['Month'] = df_combined['ds'].dt.month

        # Reorder columns
        df_combined = df_combined[['Year', 'Month', 'Prophet', 'SARIMAX', 'ARIMA', 'Random Forest', 'XGBoost', 'LSTM', 'LSTM_Adjusted']]
        # df_combined = df_combined.sort_values(by='ItemID', ascending=True)

        # Reset the index if needed
        df_combined = df_combined.reset_index(drop=True)
        # Convert the DataFrame to JSON
        df_combined_json = df_combined.to_json(orient='records', date_format='iso')

        # Print the JSON
        # print(df_combined_json)

        return df_combined_json

    except Exception as e:
        print("Exception:", str(e))  # Print any other exceptions
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)