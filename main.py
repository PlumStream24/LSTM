import pandas as pd
import numpy as np
import LSTM

df = pd.read_csv('ETH-USD-Train.csv')

X_train = np.array(df[['Open','High','Low','Close','Volume']])
y_train = np.array(df[['Open','High','Low','Close','Volume']])

def prep_data(X_datain, y_datain, time_step):
    y_indices = np.arange(start=time_step, stop=len(y_datain))
    y_tmp = y_datain[y_indices]
    
    rows_X = len(y_tmp)
    X_tmp = np.zeros((rows_X, time_step, X_datain.shape[1]))
    for i in range(X_tmp.shape[0]) :
        X_tmp[i] = X_datain[i:i+time_step]
    return X_tmp, y_tmp

timestep = 32
X_train, y_train = prep_data(X_train, y_train, timestep)

input_layer = LSTM.Input(input_shape=X_train.shape)
LSTM_layer = LSTM.LSTMLayer(prev_layer=input_layer, hidden_units=8)
flatten_layer = LSTM.Flatten(prev_layer=LSTM_layer)
dense_layer = LSTM.DenseLayer(n_neurons=5, prev_layer=flatten_layer, activation="linear")

model = LSTM.Model()
model.add_layer(LSTM_layer)
model.add_layer(flatten_layer)
model.add_layer(dense_layer)

res = model.feed_forward(X_train[0])
print(res)