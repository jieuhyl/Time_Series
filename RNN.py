# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:27:42 2020

@author: Jie.Hu
"""

import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU


df = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()


train = df.loc[:'1957-12-01'] # Goes up to but not including 108
test = df.loc['1958-01-01':]


# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


n_input = 12
n_features=1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# define model
model = Sequential()
model.add(GRU(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit_generator(generator, epochs=30)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

# forecast
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions


#test.plot(figsize=(12,8))
train['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test['Predictions'].plot(legend=True,label='PREDICTION');


# save the model
model.save('solutions_model.h5')

