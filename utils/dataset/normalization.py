from sklearn.preprocessing import MinMaxScaler
import numpy as np



class Scaler(): 
    def __init__(self):
        self.scaler = MinMaxScaler()
        pass
    
    def encode(self,series): 
        # Normalize the series using MinMaxScaler
        return self.scaler.fit_transform(series)
