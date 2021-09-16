import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from nsepy import get_history
from datetime import date

df = get_history(symbol="reliance", start=date(2010,1,1), end=date.today())
df['Date'] = df.index

plt.figure(figsize=(22,10))
plt.title('Stocks Close Price Analysis')
plt.fill_between( df['Date'], df['Close'], color="skyblue", alpha=0.5)
plt.plot(df['Date'], df['Close'], color="red", alpha=0.6)
plt.xlabel('Time/Date',fontsize=18)
plt.ylabel('Stock Close Price',fontsize=18)
plt.show()

close_col = df.filter(['Close'])
close_col_val = close_col.values


train_len = math.ceil(len(close_col_val) *.75)

mm_scale = MinMaxScaler(feature_range=(0, 1)) 
mm_scale_data = mm_scale.fit_transform(close_col_val)

train_data_val = mm_scale_data[0:train_len  , : ]

x_train=[]
y_train = []

for i in range(30, len(train_data_val)):
    x_train.append(train_data_val[i-30:i,0])
    y_train.append(train_data_val[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

network = Sequential()

network.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
network.add(Dropout(0.2))

network.add(LSTM(units = 50, return_sequences = True))
network.add(Dropout(0.2))

network.add(LSTM(units = 50, return_sequences = True))
network.add(Dropout(0.2))

network.add(LSTM(units = 50))
network.add(Dropout(0.2))

network.add(Dense(units = 1))

network.compile(optimizer = 'adam', loss = 'mean_squared_error')

network.fit(x_train, y_train, epochs = 100, batch_size = 50)



network.save('Reliance.model')

preds = network.predict(x_test) 
preds = mm_scale.inverse_transform(preds)

error_score=np.sqrt(np.mean(((preds- y_test)**2)))

training_data = close_col[:train_len]
validation_data = close_col[train_len:]
validation_data['Preds'] = preds

plt.figure(figsize=(16,8))
plt.title('LSTM Network Predicted Model')
plt.xlabel('Time/Date', fontsize=18)
plt.ylabel('Stock Close Price', fontsize=18)
plt.plot(training_data['Close'])
plt.plot(validation_data[['Close', 'Preds']])
plt.legend(['Training_values', 'Validation_values', 'Predictions_values'], loc='lower right')
plt.show()



new_close_col = df.filter(['Close'])
#Get teh last 30 day closing price 
new_close_col_val = new_close_col[100:].values
#Scale the data to be values between 0 and 1
new_close_col_val_scale = mm_scale.transform(new_close_col_val)
#Create an empty list
X_test = []
#Append teh past 1 days
X_test.append(new_close_col_val_scale)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
new_preds = network.predict(X_test)
#undo the scaling 
new_preds = mm_scale.inverse_transform(new_preds)
print(new_preds[0][0])