import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import Imputer, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder as OneHotEncoderSK
from sklearn.preprocessing import StandardScaler as StandardScalerSK

class ColumnDropper(TransformerMixin, BaseEstimator):
  
  def __init__(self, columns):
    """
    Drops selected columns from a Pandas dataframe.

    Parameters
    ==========
    columns : list of strings
      The names of the columns to be dropped.
    """
    self.columns = columns
    
  def fit(self, X, y=None):
    print('Fitting column dropper...')
    return self
  
  def transform(self, X, y=None):
    X = X.drop(self.columns, axis='columns')
    return X

  
class ConstantImputer(TransformerMixin, BaseEstimator):
  
  def __init__(self, columns, constant, null_value=np.nan):
    """
    Imputes selected columns of a Pandas dataframe with a constant
    value.

    Parameters
    ==========
    columns : list of strings
      The names of the columns to be imputed.
    constant : int or float
      The value that will replace all missing values.
    null_value : int, float, or string
      The missing value that will be replaced.
    """

    self.columns = columns
    self.constant = constant
    
  def fit(self, X, y=None):
    print('Fitting imputer...')
    return self
  
  def transform(self, X, y=None):
    X = X.copy()
    X[self.columns] = X[self.columns].fillna(self.constant)
    return X
  
  
class OneHotEncoder(TransformerMixin, BaseEstimator):
  
  def __init__(self, columns):
    """
    Applies a one-hot encoding to selected columns.

    WARNING: cannot transform new values it has not previously been fit on.

    Parameters
    ==========
    columns : list of strings
      The names of the columns to be one-hot encoded.
    """
    self.columns = columns
    self.onehot_encoder = OneHotEncoderSK(categories='auto')
  
  def fit(self, X, y=None):
    print('Fitting one-hot encoder...')
    X_onehot = X[self.columns].astype(str)
    cardinalities = X_onehot.nunique()
    self.transformed_feature_names = list()
    
    for feature, cardinality in cardinalities.iteritems():
      for i in range(cardinality):
        name = feature + f'_{i}'
        self.transformed_feature_names.append(name)
        
    self.onehot_encoder.fit(X_onehot)
    return self

  def transform(self, X, y=None):
    features = X.columns
    identity_features = set(features) - set(self.columns)
    identity_features = list(identity_features)
    
    X_identity = X[identity_features]
    X_onehot = X[self.columns].astype(str)
    X_onehot = self.onehot_encoder.transform(X_onehot).todense()
    X_onehot = pd.DataFrame(X_onehot, columns=self.transformed_feature_names)
    
    X_onehot = X_onehot.reset_index(drop=True)
    X_identity = X_identity.reset_index(drop=True)
    X = pd.concat([X_identity, X_onehot], axis='columns')
    return X
  
  
class CategoricalIndexer(TransformerMixin, BaseEstimator):
  
  def __init__(self, columns):
    """
    Applies an ordinal encoding to selected columns.

    WARNING: cannot transform new values it has not previously been fit on.

    Parameters
    ==========
    columns : list of strings
      The names of the columns to be encoded.
    """
    self.columns = columns
    self.ordinal_encoder = OrdinalEncoder()
    
  def fit(self, X, y=None):
    print('Fitting categorical indexer...')
    X_categorical = X[self.columns].astype(str)
    self.ordinal_encoder.fit(X_categorical)
    return self
  
  def transform(self, X, y=None):
    X_categorical = X[self.columns].astype(str)
    X_categorical = self.ordinal_encoder.transform(X_categorical)
    X[self.columns] = X_categorical
    return X

  
class StandardScaler(TransformerMixin, BaseEstimator):
  
  def __init__(self, columns):
    """
    Applies a standard scaling to selected columns.

    Parameters
    ==========
    columns : list of strings
      The names of the columns to be scaled.
    """
    self.columns = columns
    self.scaler = StandardScalerSK()

  def fit(self, X, y=None):
    print('Fitting standard scaler...')
    X_continuous = X[self.columns]
    self.scaler.fit(X_continuous)
    return self

  def transform(self, X, y=None):
    X_continuous = X[self.columns]
    X_continuous = self.scaler.transform(X_continuous)
    X[self.columns] = X_continuous
    return X


class TargetShifter(TransformerMixin, BaseEstimator):

  def __init__(self, target='conversion_target'):
    """
    Moves the target feature to the last column of the DataFrame
    and sorts all other columns alphabetically.

    Parameters
    ==========
    columns : str
      The name of the target feature to be moved.
    """
    self.target = target

  def fit(self, X, y=None):
    print('Fitting target shifter...')
    self.columns = sorted(list(X.columns))
    self.columns.remove(self.target)
    self.columns.append(self.target)

  def transform(self, X, y=None):
    return X[self.columns]