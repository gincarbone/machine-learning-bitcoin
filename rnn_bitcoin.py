# -*- coding: utf-8 -*-

# Recurrent Neural Network LSTM 

# Configuration Section

ai_days = 90 #we take x previous days stock prices to predict next one. Experiment with this number.
historical_prices_dataset = 'BTC_EUR_historical_gdax.csv' #same folder of the python script
last_month_dataset = 'BTC_EUR_last_month_gdax.csv' # must be 41 lines with description
neurons = 64

# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the training set
dataset_train = pd.read_csv(historical_prices_dataset)
training_set = dataset_train.iloc[:, 1:2].values

num_lines = sum(1 for line in open(historical_prices_dataset)) - 1 # remove the file first line of coloumns decription
print(num_lines)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(ai_days, num_lines): #observation , number of lines of the file
    X_train.append(training_set_scaled[i-ai_days:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = neurons, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = neurons, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = neurons, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = neurons))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#
# Wait and take a coffee while epocs make your AI more intelligent :)
#

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv(last_month_dataset)

real_data_num_lines = sum(1 for line in open(last_month_dataset)) - 1 # remove the file first line of coloumns decription
print(real_data_num_lines)

real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - ai_days:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []


for i in range(ai_days, ai_days+40):
    X_test.append(inputs[i-ai_days:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.figure(figsize=(10,5))
plt.plot(real_stock_price[10:], color = 'red', label = 'Real BTC-EUR Price')
plt.plot(predicted_stock_price[10:], color = 'grey', label = 'LSTM BTC-EUR Prediction')
plt.title('Bitcoinn Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True, linestyle='-.')
plt.show()
