from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
import numpy as np



class LabelEncoder(): 
    def __init__(self):
        self.label_encoder = OneHotEncoder()
        pass
    
    def fitNodeLable(self,df):
        # Fit the encoder on the unique labels
        self.label_encoder.fit(df[['_labels']].apply(lambda labels: [label for label in labels if label not in ['Violation', 'NotANode']]))

    def encode(self,label): 
        # Transform the labels into one-hot encoded format
        return self.label_encoder.transform(label)
    
class Tranformer(): 
    def __init__(self):
        self.tranformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        pass


    def encode(self, series):
        # Encode the text using the transformer model
        return self.tranformer.encode(series.tolist(), convert_to_numpy=True)
    
    
        

