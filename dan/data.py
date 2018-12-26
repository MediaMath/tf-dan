import os
import sys
import csv
import glob
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict

class DataGenerator:
  
  def __init__(
    self, csv_paths, vocab_map, stats_map, 
    batch_size=128, target='conversion_target', drop=['column_weights'],
    buffer_size=100000):
    """
    Generator that streams data from a set of CSVs and performs in-memory
    processing in batches, including one-hot encoding for categorical
    features and standardization for numerical features.
    
    Parameters
    ==========
    csv_paths : list of str
      Paths to each CSV that comprises the dataset. All CSVs must
      have the same header.
    vocab_map : dict
      Maps names of categorical features to lists of all possible
      unique values that feature can be (i.e. the feature's vocabulary).
    stats_map : dict
      Maps names of continuous features to dictionaries of form:
        {'mean': <mean of said feature>, 'std': <std of said feature>}.
    batch_size : int
      Number of samples in each batch.
    target : str
      Name of feature to be treated as prediction target.
    drop : list of str
      Names of features to be dropped from the dataset.
    buffer_size : int
      Window for randomly sampling data when shuffling dataset.
    """
    
    self.csv_paths = csv_paths
    if len(self.csv_paths) == 0:
      raise ValueError('csv_paths must be a non-empty list')

    self.target = target
    self.drop = drop
    self.vocab_map = vocab_map
    self.stats_map = stats_map
    print('Number of categorical columns:', len(vocab_map))
    print('Number of numerical columns:', len(stats_map))
    
    self.header = self._get_header()
    self.batch_size = batch_size
    self.num_samples = self._get_num_samples(csv_paths)
    self.num_features = self._get_num_features()
    print('Number of samples:', self.num_samples)
    
    record_defaults = self._get_record_defaults(
      self.header, vocab_map.keys(), stats_map.keys())
    self.record_defaults = record_defaults
    
    self.csv_dataset = tf.contrib.data.CsvDataset(
      filenames=self.csv_paths,
      header=True,
      record_defaults=record_defaults)
    
    self.iterator = self.csv_dataset\
      .repeat(-1)\
      .shuffle(buffer_size)\
      .batch(batch_size)\
      .make_one_shot_iterator()
    
    # Convert to list to pop individual tensors later
    self.raw_batch = list(self.iterator.get_next())
    
    # self.sess = tf.Session()
    self._create_feature_columns()
    self._create_transformation()
  
  def _get_header(self):
    """
    Checks that all CSV headers are the same and returns
    the common header.
    
    Returns
    =======
    header : list of str
      Names of each column as they appear in the dataset from
      left to right.
    """
    
    print('Checking headers for {} CSVs...'.format(len(self.csv_paths)))
    header_prev = None
    for csv_path in self.csv_paths:
      with open(csv_path, 'rt') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header_prev and header_prev != header:
          raise ValueError('CSV headers are not consistent.')
        header_prev = header
        
    print('Checking if headers consistent with maps...')
    categorical_columns = list(self.vocab_map.keys())
    numerical_columns = list(self.stats_map.keys())
    all_columns = sorted(categorical_columns + numerical_columns)
    header_sorted = sorted(header)
    if all_columns != header_sorted:
      raise ValueError('Vocab and stats maps not consistent with header.')
    return header
  
  @staticmethod
  def _get_num_samples(csv_paths):
    """
    Parameters
    ==========
    csv_paths : list of str
      Paths to each CSV that comprises the dataset.
    
    Returns
    =======
    num_samples : int
      The number of samples in the dataset.
    """
    num_samples = 0
    for csv_path in csv_paths:
      with open(csv_path, 'rt') as f:
        reader = csv.reader(f)
        num_samples += sum(1 for row in reader) - 1 # Subtract 1 for header
    return num_samples

  def _get_num_features(self):
    num_output_features = 0
    for column, vocab in self.vocab_map.items():
      if column != self.target and column not in self.drop:
        # +1 for OOV column
        num_output_features += len(set(vocab)) + 1
    for column, stats in self.stats_map.items():
      if column not in self.drop:
        num_output_features += 1
    return num_output_features
  
  @staticmethod
  def _get_record_defaults(header, categorical_columns, numerical_columns):
    """
    Parameters
    ==========
    header : list of str
      Names of all features in dataset (as they appear from left to right).
    categorical_columns : list of str
      Names of all categorical features in dataset.
    numerical_columns : list of str
      Names of all continuous features in dataset.
    
    Returns
    =======
    record_defaults : list of constant tensors, size = number of columns
      List of constant tensors, one tensor for each column, with which to
      impute each column.
    """
    if len(header) != len(categorical_columns) + len(numerical_columns):
      raise ValueError(
        'Categorical and numerical column names not consistent with header.')
    
    # Map each column name to a constant tensor for imputation
    categorical_dict = dict.fromkeys(
      categorical_columns, tf.constant(['-1'], dtype=tf.string))
    numerical_dict = dict.fromkeys(
      numerical_columns, tf.constant([0], dtype=tf.float32))
    dtype_map = {**categorical_dict, **numerical_dict}
    
    # Create list of imputation tensors consistent with order of
    # column names in the header
    record_defaults = [dtype_map[column] for column in header]
    return record_defaults
    
  def _create_feature_columns(self):
    """
    Creates a list of configured tf.feature_column objects.
    """
    feature_columns = list()
    cat_column = tf.feature_column.categorical_column_with_vocabulary_list
    for column, vocab in self.vocab_map.items():
      vocab = [str(element) for element in vocab]
      if column != self.target and column not in self.drop:
        if len(vocab) != len(set(vocab)):
          diff = len(vocab) - len(set(vocab))
          print(
            'Warning: {} duplicate terms found and removed from '
            'column {}'.format(diff, column))
        feature_column = cat_column(
          key=column, vocabulary_list=set(vocab), num_oov_buckets=1)
        feature_column = tf.feature_column.indicator_column(feature_column)
        feature_columns.append(feature_column)
      elif column == self.target:
        target_column = cat_column(
          key=column, vocabulary_list=sorted(vocab), num_oov_buckets=0)
        self.target_column = tf.feature_column.indicator_column(target_column)
      
    num_column = tf.feature_column.numeric_column
    for column, stats in self.stats_map.items():
      if column in self.drop:
        continue
      standardize = lambda x: (x - stats['mean']) / stats['std']
      feature_column = num_column(key=column, normalizer_fn=standardize)
      feature_columns.append(feature_column)
    self.feature_columns = feature_columns
    
  def _create_transformation(self):
    """
    Defines how a tensor of raw data is transformed into a tensor of
    processed data and initializes relevant variables.
    """
    # Remove the target from the header and raw_batch tensors
    target_index = self.header.index(self.target)
    self.header.remove(self.target)
    raw_batch_y = self.raw_batch.pop(target_index)
    
    # Remove drop columns from header and raw_batch tensors
    for column in self.drop:
      drop_index = self.header.index(column)
      self.header.remove(column)
      self.raw_batch.pop(drop_index)
    
    # Dictionaries with key=<feature name>, value=<raw_batch_tensor>
    features_x = dict(zip(self.header, self.raw_batch))
    features_y = {self.target: raw_batch_y}
    
    # Final tensors of processed data to be run
    # by the class' internal Tensorflow session
    self.x_processed = tf.feature_column.input_layer(
      features_x, self.feature_columns)
    self.y_processed = tf.feature_column.input_layer(
      features_y, self.target_column)

    # self.sess.run(tf.global_variables_initializer())
    # self.sess.run(tf.tables_initializer())

  # def __next__(self):
  #   """
  #   Returns a batch of processed data as a tuple with form 
  #   (batch_x, batch_y).
  #   """
  #   return self.sess.run([self.x_processed, self.y_processed])
  
  def __len__(self):
    """
    Returns the number of batches in the dataset
    (i.e. number of iterations per epoch).
    """
    return int(np.ceil(self.num_samples / self.batch_size))


def load_csv_paths(data_dir, train=True):
  csv_paths = glob.glob(os.path.join(data_dir, '*.csv'))
  if train:
    get_shard_num = lambda path: path.split('-')[-1].rstrip('.csv')
    csv_paths = sorted(csv_paths, key=get_shard_num)
  return csv_paths

def load_jsons(train_dir):
  vocab_path = os.path.join(train_dir, 'categorical-vocab.json')
  with open(vocab_path, 'rt') as f:
    vocab_map = json.load(f)

  stats_path = os.path.join(train_dir, 'numerical-stats.json')
  with open(stats_path, 'rt') as f:
    stats_map = json.load(f)

  return vocab_map, stats_map


def load_data(csv_path, target):
	"""
	Loads processed CSV data as matrices split into
	features and targets.

	Parameters
	==========
	csv_path : str
		A glob-compatible path to the CSV data.
	target : str
		Name of the target feature.

	Returns
	=======
	X : numpy array, shape = (num_samples, num_features)
		Matrix of feature values
	y : numpy array, shape = (num_samples, 2)
		One-hot matrix of target values
	"""

	paths = glob.glob(csv_path)
	dfs = [pd.read_csv(path) for path in paths]
	df = pd.concat(dfs, axis=0)

	features = list(df.columns)
	features.remove(target)

	X = df[features]
	X = X.values

	y = df[target]
	y = tf.keras.utils.to_categorical(y)

	return X, y

def load_data_imb(csv_path, target, frac=0.1, positive_upsample=1):
  """
  Loads processed CSV data as matrices split into
  features and targets.

  Parameters
  ==========
  csv_path : str
    A glob-compatible path to the CSV data.
  target : str
    Name of the target feature.

  Returns
  =======
  X : numpy array, shape = (num_samples, num_features)
    Matrix of feature values
  y : numpy array, shape = (num_samples, 2)
    One-hot matrix of target values
  """

  paths = glob.glob(csv_path)
  paths_neg = [path for path in paths if 'negative' in path]
  paths_pos = [path for path in paths if 'positive' in path]
  num_pos = int(len(paths_pos) * frac)
  paths_pos = np.random.choice(paths_pos, num_pos).tolist()
  assert(isinstance(positive_upsample, int))
  paths = paths_neg + paths_pos * positive_upsample
  dfs = [pd.read_csv(path) for path in paths]
  df = pd.concat(dfs, axis=0)

  features = list(df.columns)
  features.remove(target)

  X = df[features]
  X = X.values

  y = df[target]
  y = tf.keras.utils.to_categorical(y)

  return X, y


def load_cardinality_map(json_path):
	with open(json_path, 'rt') as f:
		index_cardinality_map = json.load(f)

		# Convert keys from strings to integers
		index_cardinality_map = {\
			int(key): value\
			for key, value\
			in index_cardinality_map.items()}
	return index_cardinality_map
