import glob
import os
import gzip
import csv
import json

import numpy as np
import pandas as pd
import transformers as tr
import argparse as ap

from sklearn.pipeline import Pipeline

parser = ap.ArgumentParser()
parser.add_argument(
	'--cardinality_threshold', type=int, default=20,
	help=(
		'Categorical features with cardinalities above this threshold '
		'are index encoded. Otherwise, they are one-hot encoded.'))
parser.add_argument(
	'--train_source', type=str,
	default='../data/raw/mm-ctr/train/data-*/*.csv.gz',
	help='Paths to training data. Supports globbing.')
parser.add_argument(
	'--validation_source', type=str,
	default=(
		'../data/raw/mm-ctr/validation/'
		'*.csv.gz'),
	help='Path to validation data. Supports globbing.')
parser.add_argument(
	'--test_source', type=str,
	default=(
		'../data/raw/mm-ctr/test/'
		'*.csv.gz'),
	help='Path to test data. Supports globbing.')
parser.add_argument(
	'--destination_dir', type=str, default='../data/processed/mm-ctr/',
	help='Directory where processed CSVs are saved.')
parser.add_argument(
	'--intermediate_dir', type=str, default='../data/intermediate/mm-ctr/',
	help='Directory where intermediate CSVs are saved.')
parser.add_argument(
	'--overwrite', action='store_true',
	help='Overwrite already existing processed CSVs.')

args = parser.parse_args()
CARDINALITY_THRESHOLD = args.cardinality_threshold
TRAIN_SOURCE = args.train_source
VALIDATION_SOURCE = args.validation_source
TEST_SOURCE = args.test_source
DESTINATION_DIR = args.destination_dir
INTERMEDIATE_DIR = args.intermediate_dir
OVERWRITE = args.overwrite

# Perform directory checks and creation
train_pos_csv_name = 'train-positive.csv'
train_neg_csv_name = 'train-negative.csv'
validation_csv_name = 'validation.csv'
test_csv_name = 'test.csv'

csv_names = [
	train_pos_csv_name,
	train_neg_csv_name,
	validation_csv_name,
	test_csv_name,
]

# csv_paths = [os.path.join(DESTINATION_DIR, name) for name in csv_names]
# path_exists = [os.path.isfile(csv_path) for csv_path in csv_paths]

# if all(path_exists) and not OVERWRITE:
# 	print('Processed files already exist.')
# 	exit()

if not os.path.exists(DESTINATION_DIR):
	print(f'Creating new directory: {DESTINATION_DIR}')
	os.makedirs(DESTINATION_DIR)

if not os.path.exists(INTERMEDIATE_DIR):
	print(f'Creating new directory: {INTERMEDIATE_DIR}')
	os.makedirs(INTERMEDIATE_DIR)


# Read training data
train_csv_paths = glob.glob(TRAIN_SOURCE)
val_csv_paths = glob.glob(VALIDATION_SOURCE)
test_csv_paths = glob.glob(TEST_SOURCE)
all_csv_paths = train_csv_paths + val_csv_paths + test_csv_paths
print(f'Found {len(all_csv_paths)} CSV files.')

all_dfs = [pd.read_csv(path) for path in all_csv_paths]
all_df = pd.concat(all_dfs, axis='rows', join='inner')
print(
	'All data contains {} rows '
	'and {} features.'.format(*all_df.shape))

# Divide all features into 4 mutually exclusive groups:
# - drop_features: features to be dropped
# - continuous_features: features that will be imputed and undergo scaling
# - onehot_features: features that will be imputed and undergo one-hot 
#		encoding
# - index_features: features that will be imputed and undergo index encoding

cardinalities = all_df.nunique()
drop_features = set(cardinalities[cardinalities == 1].index)
drop_features = drop_features.union(set([
	'column_weights', ]))
    #'mm_1211339_pixel', 'mm_1211339_bpr', 'mm_1211339_bpf'

# Need for populating continuous features later
all_features_index = cardinalities.index

all_features = set(all_features_index)
all_features = all_features - drop_features

# Defined by Aravind
categorical_features = {
	'exchange_id', 'user_frequency', 'site_id', 'deal_id',
	'channel_type', 'size', 'week_part', 'day_of_week', 'dma_id', 'isp_id',
	'fold_position', 'browser_language_id', 'country_id', 'conn_speed', 'os_id',
	'day_part', 'region_id', 'browser_id', 'hashed_app_id', 'interstitial',
	'device_id', 'creative_id', 'browser', 'browser_version', 'os', 
	'os_version', 'device_model', 'device_manufacturer', 'device_type',
	'exchange_id_cs_vcr', 'exchange_id_cs_vrate', 'exchange_id_cs_ctr', 
	'exchange_id_cs_category_id', 'exchange_id_cs_site_id','category_id', 
	'cookieless', 'cross_device'}
categorical_features = categorical_features - drop_features

# Defined by Aravind
continuous_features = [
	'id_vintage', 'exchange_viewability_rate', 'exchange_ctr', 'exchange_vcr']
continuous_features += list(
	all_features_index[all_features_index.str.contains('_bpr')])
continuous_features += list(
	all_features_index[all_features_index.str.contains('_bpf')])
continuous_features += list(
	all_features_index[all_features_index.str.contains('pixel')])
continuous_features = set(continuous_features) - drop_features

num_features_1 = len(cardinalities.index) - 1 # Do not count target
num_features_2 = len(drop_features) \
	+ len(categorical_features) \
	+ len(continuous_features)

#chg
#print(num_features_1)
#print(num_features_2)
assert(num_features_1 == num_features_2)

# Split categorical features into ones that will be one-hot encoded
# and index encoded
onehot_features = set(
	cardinalities[cardinalities <= CARDINALITY_THRESHOLD].index)
onehot_features = onehot_features.intersection(categorical_features)
index_features = categorical_features - onehot_features

#print (index_features)

num_features_2 = len(drop_features) \
	+ len(continuous_features) \
	+ len(onehot_features) \
	+ len(index_features)

#chg
assert(num_features_1 == num_features_2)
assert(len(categorical_features) == \
			 len(onehot_features) + len(index_features))

# Convert all sets to lists
categorical_features = list(categorical_features)
continuous_features = list(continuous_features)
onehot_features = list(onehot_features)
index_features = list(index_features)
drop_features = list(drop_features)

print(f'Found {len(drop_features)} features to be dropped.')
print(f'Found {len(onehot_features)} features to be one-hot encoded.')
print(f'Found {len(index_features)} features to be ordinal encoded.')
print(f'Found {len(continuous_features)} features to be scaled.')

# Build pipeline
column_dropper = tr.ColumnDropper(
	columns=drop_features)
constant_imputer_cat = tr.ConstantImputer(
	columns=categorical_features, constant=-1)
constant_imputer_cont = tr.ConstantImputer(
	columns=continuous_features, constant=0)
categorical_indexer = tr.CategoricalIndexer(
	columns=index_features)
scaler = tr.StandardScaler(
	columns=continuous_features)
onehot_encoder = tr.OneHotEncoder(
	columns=onehot_features)
target_shifter = tr.TargetShifter(
	target='conversion_target')

pipeline = Pipeline([
	('column_dropper', column_dropper),
	('constant_imputer_cat', constant_imputer_cat),
	('constant_imputer_cont', constant_imputer_cont),
	('categorical_indexer', categorical_indexer),
	('onehot_encoder', onehot_encoder),
	('target_shifter', target_shifter),
])

# Will take about 5 minutes to fit...
pipeline.fit(all_df)
del all_df

train_dfs = [pd.read_csv(path) for path in train_csv_paths]
train_df = pd.concat(train_dfs, axis='rows')
print(
	'Training data contains {} rows '
	'and {} features.'.format(*train_df.shape))
scaler.fit(train_df)
del train_df

def process_data(source, destination, pipeline):
	if isinstance(source, str):
		source = [source]
	with open(destination, 'w') as file_out:
		writer = csv.writer(file_out)
		header = None
		for src in source:
			df = pd.read_csv(src)
			df = scaler.transform(df)
			df_transformed = pipeline.transform(df)
			if header is None:
				header = df_transformed.columns
				writer.writerow(header)

				# Create a JSON with column index of <index_features> as keys
				# and cardinality as value. Will be used to setup embedding layers.
				index_cardinality_map = dict()
				for index_feature in index_features:
					index = header.get_loc(index_feature)
					cardinality = int(cardinalities[index_feature]) + 1
					index_cardinality_map[index] = cardinality

				json_path = os.path.join(DESTINATION_DIR, 'index-cardinality.json')
				with open(json_path, 'w') as f:
					json.dump(index_cardinality_map, f)

			for index, row in df_transformed.iterrows():
				writer.writerow(row.values)

print('Processing validation data...')
process_data(
	val_csv_paths,
	os.path.join(DESTINATION_DIR, validation_csv_name),
	pipeline,
)

print('Processing test data...')
process_data(
	test_csv_paths,
	os.path.join(DESTINATION_DIR, test_csv_name),
	pipeline,
)

print('Processing positive training data...')
train_pos_csv_paths = [path for path in train_csv_paths if 'positive' in path]
process_data(
	train_pos_csv_paths,
	os.path.join(INTERMEDIATE_DIR, train_pos_csv_name),
	pipeline)

print('Processing negative training data...')
train_neg_csv_paths = [path for path in train_csv_paths if 'negative' in path]
process_data(
	train_neg_csv_paths,
	os.path.join(INTERMEDIATE_DIR, train_neg_csv_name),
	pipeline)
