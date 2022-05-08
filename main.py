import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import math

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
start_date = st.sidebar.date_input("Start Date", datetime.date(2012,1,1))
end_date = datetime.date.today()

stocks = ('AAPL','GOOGL', 'MSFT', 'LNVGY', 'AMZN','INTC')

tickerSymbol = st.sidebar.selectbox('Stock Ticker',stocks)
tickerData = yf.Ticker(tickerSymbol) #get ticker data
tickerDf = tickerData.history(period = '1d', start_date, end_date)

st.subheader('Ticker Data ') 
st.write(tickerDf)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Close'],  name='Stock Close'))
    fig.update_layout(title="This is the Trend in the Raw Data")
    st.plotly_chart(fig)
  
  
plot_raw_data()
data_close = tickerDf['Close']
scaler = MinMaxScaler(feature_range = (0,1))
data_close = scaler.fit_transform(np.array(data_close).reshape(-1,1))

#Split the data into train and test split
training_size = int(len(data_close)*0.65)
test_size = len(data_close)-training_size
train_data, test_data = data_close[0:training_size,:], data_close[training_size:len(data_close)]

@st.cache
def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)
                     
time_step = 200
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
                     
#reshape the input to be [sample, time steps, features] which is the requirement of LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                     
#Create the LSTM model
model = Sequential() 
model.add(LSTM(50 ,return_sequences = True, input_shape = (200,1)))
model.add(LSTM(50, return_sequences= True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()   

model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=4, batch_size=70, verbose=1)

#Lets predict and check performance metrics
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

#Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train, train_predict))
#Plotting

#Shift train prediction for plotting
look_back = 200
trainPredictPlot = np.empty_like(data_close)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#Shift test prediction for plotting
testPredictPlot = np.empty_like(data_close)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2)+1:len(data_close) - 1, :] = test_predict
itdc = pd.DataFrame(scaler.inverse_transform(data_close))

#Plot baseline and predictions
tpp = pd.DataFrame(trainPredictPlot)

tepp = pd.DataFrame(testPredictPlot)

agree= st.checkbox('Want to see How the Data is splitted ? ')
if agree:
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=data['Date'], y=itdc[0], name='Closing Price'))
   fig.add_trace(go.Scatter(x=data['Date'], y=tpp[0], name='Train Predict'))
   fig.add_trace(go.Scatter(x=data['Date'], y=tepp[0], name='Test Predict'))
   st.plotly_chart(fig)

x_input = test_data[len(test_data)-200:].reshape(-1,1,1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=200
i=0
while(i<30):
    
    if(len(temp_input)>200):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
       
        x_input=x_input.reshape(1,-1,1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    
day_new = pd.DataFrame(np.arange(1,201))
day_pred = pd.DataFrame(np.arange(201,231))

dp1 = pd.DataFrame(scaler.inverse_transform(data_close[(len(data_close)-200):]))
dp2 = pd.DataFrame(scaler.inverse_transform(lst_output))

st.write('Hey ! Look what our LSTM  Predicted ....Just a min')

fig = go.Figure()
fig.add_trace(go.Scatter(x=day_new[0], y=dp1[0],  name='Prev 100 days Test data '))
fig.add_trace(go.Scatter(x=day_pred[0], y=dp2[0], name='30 days predict'))
st.plotly_chart(fig)

df1=data_close.tolist()
df1.extend(lst_output)

st.write('More clearer way for Analyzing The Results.' )

df1=scaler.inverse_transform(df1).tolist()

st.line_chart(df1)
         
st.write('And Done!!!!  ')
