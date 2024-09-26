import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from sklearn.neural_network import MLPRegressor

class RNN_regression:

    def __init__(
            self,
            input_length : int = 10,
            input_dim : int = 7,
            output_dim : int = 1,
            ) -> None:
             
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(input_length, input_dim)))
        # self.model.add(Normalization())
        self.model.add(LSTM(units=128, return_sequences=False))
        # self.model.add(LSTM(units=64, return_sequences=False))
        self.model.add(Dense(output_dim, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        # self.model.summary()

    def get_model(self):
        return self.model
    
    def fit(self,X,Y):
        self.model.fit(X,Y,epochs=20,batch_size=16, shuffle=True)

    def predict(self,X):
        return self.model.predict(X)
    
    def save(self, path : str):
        self.model.save(path)    
    
class MLP_regression:

        def __init__(
            self,
            input_length : int = 10,
            input_dim : int = 7,
            output_dim : int = 1,
            ) -> None:
             
            self.model = MLPRegressor(
                hidden_layer_sizes=(100,100),
                )

        def get_model(self):
            return self.model
        
        def fit(self,X,Y):
            X = X.reshape((len(X),-1))
            if Y.shape[1] == 1:
                Y = Y.ravel()
            else : 
                Y = Y.reshape((len(Y),-1))
            self.model.fit(X,Y)

        def predict(self,X):            
            X = X.reshape((len(X),-1))
            return self.model.predict(X)   

        def save(self, path : str):
            print("Can't save sklearn model")

        def score(self,X,Y):
            X = X.reshape((len(X),-1))
            if Y.shape[1] == 1:
                Y = Y.ravel()
            else : 
                Y = Y.reshape((len(Y),-1))
            return self.model.score(X,Y)        

