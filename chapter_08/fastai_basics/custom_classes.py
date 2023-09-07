from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np





class encode_categorical(BaseEstimator, TransformerMixin):
    '''encode categorical columns'''
    def __init__(self):
        self.le = {}
        self.max_dict = {}
        return None
    
    def set_params(self, **kwargs):
        self.col_list = kwargs.get('col_list', None)
        return self
        
    
    
    def fit(self, X, y=None,  **fit_params):
        for col in self.col_list:
            print("col is ",col)
            self.le[col] = LabelEncoder()
            self.le[col].fit(X[col].tolist())
        return self
        
    
    def transform(self, X, y=None, **tranform_params):
        for col in self.col_list:
            print("transform col is ",col)
            X[col] = self.le[col].transform(X[col])
            print("after transform col is ",col)
            self.max_dict[col] = X[col].max() +1
        # print(X.loc[[0]])
        return X

class prep_for_keras_input(BaseEstimator, TransformerMixin):
    '''prep columns for feeding Keras model'''
    def __init__(self):
        self.dictlist = []
        return None
    
    def set_params(self, **kwargs):
        self.collist = kwargs.get('collist', None)
        self.continuouscols = kwargs.get('continuouscols', None)
        self.textcols = kwargs.get('textcols', None)
        return self
        
    
    
    def fit(self, X, y=None,  **fit_params):
        return self
        
    
    def transform(self, X, y=None, **tranform_params):
        for col in self.collist:
            print("cat col is",col)
            self.dictlist.append(np.array(X[col]))
        for col in self.textcols:
            print("text col is",col)
            self.dictlist.append(pad_sequences(X[col], maxlen=max_dict[col]))
        for col in self.continuouscols:
            print("cont col is",col)
            self.dictlist.append(np.array(X[col]))    
        # print(X.loc[[0]])
        return self.dictlist

class fill_empty(BaseEstimator, TransformerMixin):
    '''fill empty values with placeholders'''
    
    
    def set_params(self, **kwargs):
        self.collist = kwargs.get('collist', None)
        self.continuouscols = kwargs.get('continuouscols', None)
        self.textcols = kwargs.get('textcols', None)
        return self
    
    
    def transform(self, X, **tranform_params):
        print("fill empty xform")
        for col in self.collist:
            X[col].fillna(value="missing", inplace=True)
        for col in self.continuouscols:
            X[col].fillna(value=0.0,inplace=True)
        for col in self.textcols:
            X[col].fillna(value="missing", inplace=True)
        return X
    
    def fit(self, X, y=None, **fit_params):
        return self

class encode_text(BaseEstimator, TransformerMixin):
    '''encode text columns'''
    def __init__(self):
        self.tok = {}
        return None
    
    def set_params(self, **kwargs):
        self.col_list = kwargs.get('col_list', None)
        return self
        
    
    
    def fit(self, X, y=None,  **fit_params):
        for col in self.col_list:
            print("col is ",col)
            self.tok[col] = Tokenizer(num_words=maxwords,lower=True)
            self.tok[col].fit_on_texts(X[col])
        return self
        
    
    def transform(self, X, y=None, **tranform_params):
        for col in self.col_list:
            print("transform col is ",col)
            X[col] = self.tok[col].texts_to_sequences(X[col])
            print("after transform col is ",col)
            self.max_dict[col] = np.max(X[(X[col].map(len) != 0)][col].map(max))
            if self.max_dict[cols] > textmax:
                textmax = self.max_dict[cols]
        # print(X.loc[[0]])
        return X
