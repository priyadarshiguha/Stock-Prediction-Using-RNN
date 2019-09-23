import numpy as np
import pandas as pd
import matplotlib.pyplot as graph
from keras.callbacks import ModelCheckpoint

data_train=pd.read_csv('Google_Stock_Price_Train.csv')
train_set=data_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler(feature_range=(0,1))
train_set_scaled=scale.fit_transform(train_set)

seq_len=60
X_Train=[]
Y_Train=[]
for i in range(0, len(train_set_scaled)-seq_len):
    X_Train.append(train_set_scaled[i:i+seq_len, 0])
    Y_Train.append(train_set_scaled[i+seq_len, 0])
X_Train, Y_Train=np.array(X_Train), np.array(Y_Train)

X_Train=np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], 1))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

RNN_Model=Sequential()

#layer 1
RNN_Model.add(LSTM(units=100, return_sequences=True, input_shape=X_Train[0].shape))
RNN_Model.add(Dropout(0.2))
#layer 2
RNN_Model.add(LSTM(units=100, return_sequences=True))
RNN_Model.add(Dropout(0.2))
#layer 3
RNN_Model.add(LSTM(units=100, return_sequences=True))
RNN_Model.add(Dropout(0.2))
#layer 4
RNN_Model.add(LSTM(units=100, return_sequences=True))
RNN_Model.add(Dropout(0.2))

RNN_Model.add(Flatten())

#output layer
RNN_Model.add(Dense(units=1))

RNN_Model.compile(optimizer='adam', loss='mean_squared_error')

filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#fit the model
RNN_Model.fit(X_Train, Y_Train, epochs=100, batch_size=32, callbacks=callbacks_list)

#load the model with saved weights
filename = "weights.hdf5"
RNN_Model.load_weights(filename)
RNN_Model.compile(loss='mean_squared_error', optimizer='adam')

data_test=pd.read_csv('Google_Stock_Price_Test.csv')
test_set=data_test.iloc[:, 1:2].values

data_total=pd.concat((data_train['Open'], data_test['Open']), axis=0)
inputs=data_total[len(data_total)-len(data_test)-seq_len:].values
inputs=inputs.reshape(-1,1)
inputs=scale.transform(inputs)
X_Test=[]
for i in range(0, len(data_test)):
    X_Test.append(inputs[i:i+seq_len, 0])
X_Test=np.array(X_Test)
X_Test=np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1], 1))
predicted_stock=RNN_Model.predict(X_Test)
predicted_stock=scale.inverse_transform(predicted_stock)

graph.plot(test_set, color='red')
graph.plot(predicted_stock, color='green')
graph.title('Stock Prediction')
graph.xlabel('Time')
graph.ylabel('Stock')
graph.show()