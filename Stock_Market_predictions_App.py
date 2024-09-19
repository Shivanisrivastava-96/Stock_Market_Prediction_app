# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 00:13:14 2024

@author: shiva
"""
# Import required libraries
import sys
sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K    
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Title
app_name = "Stock Market Prediction App using ARIMA and LSTM"
st.title(app_name)
st.subheader("This app is created to forecast the stock market price of the selected company.")

# add an image from online resourse 
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from the user of app about start & end date 

#sidebar 
st.sidebar.header("Select the Parameter from below")

# symbol list
ticker_list = ["AAPL","INFY","MSFT","GOOGL","TSLA","NVDA"]

# Dropdown list
ticker=st.sidebar.selectbox("Select the company",ticker_list)

start_date=st.sidebar.date_input('Start date',pd.to_datetime('2014-01-01'))
end_date=st.sidebar.date_input('End date',pd.to_datetime('2024-01-01'))

# fetch data from user inputs yfinance library 
data=yf.download(ticker,start=start_date,end=end_date)

# Display the data 
st.subheader(f'Stock price data for {ticker}')
st.write(data)

# Create the 'Volume (Thousands)' column by dividing the original 'Volume' by 1,000
data['Volume'] = data['Volume'] / 1_000

# Add date as column to the dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write("Date from",start_date,"To",end_date)
# Display the filtered data
st.write(data)
# Plot the data
st.subheader('Stock Price Plot')
fig=px.line(data,x="Date",y=data.columns,title="Closing price of the stock",width=800,height=800)
st.plotly_chart(fig)

# selecting the model
model_choice=st.sidebar.selectbox("Choose a model",["ARIMA","LSTM"])

# Add a selectbox to select column from the data 
column=st.sidebar.selectbox("Select the column to be used for forecasting",data.columns[1:])

# Subsetting the data for selecting date vs column 
Data=data[["Date",column]]
st.write("Selected Data",Data)

# ADF Test to check stationarity
st.subheader('ADF Test for Stationarity')
adf_result = adfuller(data[column])
st.write(f"ADF Statistic: {adf_result[0]}")
st.write(f"p-value: {adf_result[1]}")
st.write('Data is stationary' if adf_result[1] < 0.05 else 'Data is not stationary')

# Decompose the data 
st.header("Decompostion of data")
decomposition=seasonal_decompose(data[column],model="multiplicative",period=12) 
st.write(decomposition.plot())

# Using Sarimax  first trained the model
def arima_model(data,p,d,q,seasonal_order):
    
    
    # Using Sarimax  first trained the model
    model_1= sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))

   # fit the model
    model_1=model_1.fit()
           
   # Print the summary of the model 
    st.header("Model_Summary")
    st.write(model_1.summary())

   # Predict the future values (forecasting)
    forecast_period=st.number_input("Select the number of days to forecast",1,365,10)

    predictions=model_1.get_forecast(steps=forecast_period)

   # Get confidence intervals for predictions
    pred_ci = predictions.conf_int()

   # Extract the Predicted mean values (forecasted values)
    predicted_mean=predictions.predicted_mean

   # Display the predicted values and the confidence intervals
    st.write("Confidence Interval:-")
    st.write(pred_ci)

   # Change the index of predictions column 
    predicted_mean.index=pd.date_range(start=end_date,periods=forecast_period,freq="D")

   # Convert to Dataframe for better display
    predicted_mean_df=pd.DataFrame(predicted_mean)

   # add the date into index
    predicted_mean_df.insert(0, "Date",predicted_mean.index,True)
    predicted_mean_df.reset_index(drop=True,inplace=True)
    st.write("Predicted Data",predicted_mean_df)
    st.write("Actual Data",Data)
    st.write("----")

   # Plot the Actual & Predicted data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode="lines",name="Actual_Data",line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=predicted_mean_df["Date"],y=predicted_mean_df["predicted_mean"],mode="lines",name="Predicted_Data",line=dict(color="red")))

   # Add the title
    fig.update_layout(title="Actual vs Predicted",xaxis_title="Date",yaxis_title="Price")

   # display the plot 
    st.plotly_chart(fig)

   # Calculate the value of RMSE & MSE 
    actual_values = data[column].iloc[-forecast_period:].values  # Last n actual values
    predicted_values = predictions.predicted_mean.values  # The forecasted values

   # Now calculate RMSE
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)

   # Display RMSE in Streamlit
    st.write(f'Mean Square Error(MSE),{mse:.4f}')
    st.write(f'Root Mean Square Error(RMSE),{rmse:.4f}')

# Using LSTM model 
# Scale data (LSTM performs better with scaled data)

def lstm_model(data):
    K.clear_session()  # Clear previous models from memory
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(data[column].values.reshape(-1,1))
    
    # prepare the data
    x_train=[]
    y_train=[]
    for i in range(60,len(data)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
        
# i-60:i: This specifies a range of rows. It means, "from the i-60-th row up to the i-th row (but not including i)." The value 60 is the look-back window size, i.e., we are looking at the last 60 time steps to predict the next time step.

# 0: This selects the first column of the array. Since scaled_data has only one column (representing the stock's closing price), this index is used to access the first and only column of the data.
        
    # reshape the x_train & y_train 
    x_train,y_train = np.array(x_train),np.array(y_train)
    
    # reshape the data to match LSTM's expected input format
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
# Build LSTM model 
    model=Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))   # For regression (predicting one value)      
# (number of samples, number of time steps, number of features)
    model.compile(optimizer="adam",loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=20, batch_size=32)
    
# Predict 
    predicted_data=model.predict(x_train)
    predicted_data=scaler.inverse_transform(predicted_data)
    
    return predicted_data

# Run the selected model
if model_choice == 'LSTM':
    lstm_pred = lstm_model(data)
    st.subheader('LSTM Predictions')
    fig, ax = plt.subplots()
    ax.plot(data.index, data[column], label='Original')
    ax.plot(data.index[60:], lstm_pred, label='Predicted', color='red')
    ax.set_title("Price Prediction using LSTM")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

elif model_choice == 'ARIMA':
    
    # Add the slider for selecting values of (p,d,q)
    p=st.sidebar.slider("Select the value of p",0,5,2)
    d=st.sidebar.slider("Select the value of d",0,2,1)
    q=st.sidebar.slider("Select the value of q",0,5,2)
    seasonal_order=st.sidebar.number_input("Select the value of sp:-",1,24,12)
    st.subheader('ARIMA Predictions')
    forecast = arima_model(data,p,d,q,seasonal_order)
    st.line_chart(forecast) 