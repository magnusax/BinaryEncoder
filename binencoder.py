import bisect
import numpy as np
import pandas as pd
from numbers import Number
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder


class BinaryEncoder(TransformerMixin):
    """ Binary encoder class 

    Parameters:
    -----------
        sentinel : integer, float, string (optional)
            Provide a sentinel value to use when labelling
            future unseen values.
            - Note that the sentinel `dtype` should match
              the dtype of data that is used as input to
              the transformation.
    """ 

    def __init__(self, sentinel=None):
        self.le_ = None        
        self.col_ = None
        self.num_output_columns_ = None
        if sentinel:
            self.sentinel_ = sentinel
        
    def __repr__(self):
        return "BinaryEncoder(sentinel=None)"        
        
    def _set_sentinel(self, classes):
        value = classes[0]
        if isinstance(value, (Number, np.number)):
            sentinel = -99999
        elif isinstance(value, str):
            sentinel = '<unknown>'
        return sentinel
    
    def _compute_series(self, X):
        return pd.Series(self.le_.transform(X), name=self.col_)

    def fit(self, X, y=None):        
        self.col_ = X.name
        self.le_ = LabelEncoder()
        self.le_.fit(X)

        # Handle future cases with unseen values
        classes = self.le_.classes_.tolist()
        if self.sentinel_ is not None:
            self.sentinel_ = self._set_sentinel(classes)

        # Handles insertion while maintaining order
        bisect.insort_left(classes, self.sentinel_)
        self.le_.classes_ = classes

        series = self._compute_series(X)
        self.num_output_columns_ = int(np.ceil(np.log2(1+max(series))))
        
        return self        
        
    def transform(self, X):     
        # Map unseen values to sentinel value
        X = X.map(lambda s: self.sentinel_ if s not in self.le_.classes_ else s)
        
        # Compute binary encoding
        series = self._compute_series(X)
        series = series.apply(lambda x: np.binary_repr(x, width=self.num_output_columns_))
        df = series.to_frame()
        df.columns = [self.col_]
        
        # Convert encoding to suitably formatted columns
        for c in range(self.num_output_columns_):
            new_col = "".join((self.col_, "_", str(c))) 
            df[new_col] = df[self.col_].apply(lambda e: list(e)[c]).astype('category')
        df = df.drop(self.col_, axis=1)
        return df
