import os
import pickle
import warnings
import numpy as np
import streamlit as st
from datetime import date
import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model

st.title('Stock Predictor')

st.write('Shown are the stock price data for query companies!')
st.markdown('''
**Credits**
- App built by Nehal, Sara, Alok
- Built in `Python` using `streamlit`,`yfinance`, `pandas` and `datetime`
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Choose Your Query Parameter ')
start_date = st.sidebar.date_input("Start Date", date(1999,1,1))
end_date = date.today().strftime("%Y-%m-%d")

##### Dropdown to choose ticker #####
ticker = st.selectbox("Choose ticker :", ['AAPL [Apple]'
                                 #'FB [Facebook]',
                                 #'HDB [HDFC Bank Limited]',
                                 #'MSFT [Microsoft]',
                                 #'TSLA [Tesla, Inc.]'
                                ])

# Extract stock ticker
stock_ticker = ticker.split(' ')[0]
stock_comp = ticker.split(' ')[1]
stock_comp = stock_comp[1:-1]


plot_df = web.DataReader(stock_ticker, data_source='yahoo', start=start_date, end=end_date)
st.write(plot_df)


##### Displaying `Close` Price chart #####  
fig = plt.figure(figsize=(12,5))
plt.title(stock_comp + " - Stock Price History")
plt.plot(plot_df['Close'])
plt.xlabel("Years")
plt.ylabel("Close Price in USD($)")
st.pyplot(fig)

# Insert empty line
st.progress(100)

nb_days = st.selectbox("Get Stock Price Forecast for :", ['1 Day', '2 Days', '3 Days'])

# define a placeholder
ph = st.empty()

# Logger function for UI
def logger(ph, message):
    if message == "":
        ph.write(message)
    else:
        ph.write("[INFO] " + message + "...")

# Function to get forecast after button click
def getForecast(ticker, nb_days):
    
    # Get the stock price data and store in temp datagframe
    curr_date = str(date.today())
    df = web.DataReader(stock_ticker, data_source='yahoo', start='2010-01-01', end=curr_date)
    temp_df = df.iloc[len(df)-90 : len(df), :]
    
    
    # get 'Close' price and covert to numpy array
    user_data = temp_df['Close'].values
    
    
    # Load model and scaler
    model_path, scaler_path = "", ""
    cwd = os.getcwd()
    if ticker == 'AAPL':
        if nb_days == '1 Day':
            # NOTE : COMMENTED PATH WORK ON LOCAL SYSTEM AND UNCOMMENTED ONES WORK FOR WEB APP
            #model_path = cwd + '\\Apple\\AAPL_1_day_SPF_model'
            #scaler_path = cwd + '\\Apple\\AAPL_1_day_SPF_scaler.pkl'
            model_path = cwd + '/Apple/AAPL_1_day_SPF_model'
            scaler_path = cwd + '/Apple/AAPL_1_day_SPF_scaler.pkl'
        elif nb_days == '2 Days':
            #model_path = cwd + '\\Apple\\AAPL_2_days_SPF_model'
            #scaler_path = cwd + '\\Apple\\AAPL_2_days_SPF_scaler.pkl'
            model_path = cwd + '/Apple/AAPL_2_days_SPF_model'
            scaler_path = cwd + '/Apple/AAPL_2_days_SPF_scaler.pkl'
        else:
            #model_path = cwd + '\\Apple\\AAPL_3_days_SPF_model'
            #scaler_path = cwd + '\\Apple\\AAPL_3_days_SPF_scaler.pkl'
            model_path = cwd + '/Apple/AAPL_3_days_SPF_model'
            scaler_path = cwd + '/Apple/AAPL_3_days_SPF_scaler.pkl'
    logger(ph, "Loading saved model")
    model = load_model(model_path)
    logger(ph, "Loading saved scaler")
    f = open(scaler_path, 'rb')
    scaler = pickle.load(f)
    
    
    # Scale the data 
    user_data = user_data.reshape(1, user_data.shape[0])
    user_data = scaler.transform(user_data)
    
    
    # Convert shape and form of data that is accepted by LSTM RNN
    user_data = np.reshape(user_data, (user_data.shape[0], user_data.shape[1], 1))
    
    
    # Predict next day stock price
    logger(ph, "Predicting stock price(s)")
    prediction = model.predict(user_data)
    print(prediction)
    return prediction

##### Button to get desired forecast #####
if st.button("Get Forecast"):
    priceForecast = getForecast(stock_ticker, nb_days)
    logger(ph, "")
    if nb_days == '1 Day':
        priceForecastDay1 = round(priceForecast[0][0], 4)
        st.success("Next " + nb_days + " stock price forecast (in $) :\n\n " + 
                   "**Day 1** : " + str(priceForecastDay1))
    elif nb_days == '2 Days':
        priceForecastDay1 = round(priceForecast[0][0], 4)
        priceForecastDay2 = round(priceForecast[0][1], 4)
        st.success("Next " + nb_days + " stock price forecast (in $) :\n\n" + 
                   "**Day 1** : " + str(priceForecastDay1) + "\n\n" 
                   "**Day 2** : " + str(priceForecastDay2))
    else:
        priceForecastDay1 = round(priceForecast[0][0], 4)
        priceForecastDay2 = round(priceForecast[0][1], 4)
        priceForecastDay3 = round(priceForecast[0][2], 4)
        st.success("Next " + nb_days + " stock price forecast (in $) :\n\n" + 
                   "**Day 1** : " + str(priceForecastDay1) + "\n\n" +  
                   "**Day 2** : " + str(priceForecastDay2) + "\n\n" + 
                    "**Day 3** : " + str(priceForecastDay3))


