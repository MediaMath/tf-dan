import os
import sys
import glob
import json
import pytest

import pandas as pd
import numpy as np

sys.path.append('..')

from dan.data import load_data, load_csv_paths, load_jsons, DataGenerator

TRAIN_DIR = '../data/processed/mm-cpc-generator/train/'
TARGET = 'conversion_target'
BATCH_SIZE = 128

@pytest.fixture()
def make_datagen():
	csv_paths = load_csv_paths(TRAIN_DIR, train=True)[:2]
	vocab_map, stats_map = load_jsons(TRAIN_DIR)
	datagen = DataGenerator(
		csv_paths, vocab_map, stats_map,
		batch_size=BATCH_SIZE,
		target='conversion_target',
		drop=['column_weights'],
		buffer_size=1000)
	return datagen

def test_load_data():
	path = os.path.join(TRAIN_DIR, 'train-*-0000.csv')
	X_train, y_train = load_data(path, TARGET)
	assert(len(X_train.shape) == 2)
	assert(len(y_train.shape) == 2)
	assert(X_train.shape[0] == y_train.shape[0])
	assert(y_train.shape[1] == 2)
	assert(X_train.shape[1] == 96)

def test_generator_init(make_datagen):
	datagen = make_datagen

def test_generator_len(make_datagen):
	csv_paths = load_csv_paths(TRAIN_DIR, train=True)[0]
	num_samples = pd.read_csv(csv_paths).shape[0] * 2
	datagen = make_datagen
	assert(datagen.num_samples == num_samples)
	assert(len(datagen) == np.ceil(num_samples / datagen.batch_size))

def test_generator_next(make_datagen):
	datagen = make_datagen
	for i in range(len(datagen)):
		batch_x, batch_y = next(datagen)
		assert(batch_x.shape[0] == BATCH_SIZE)
		assert(batch_y.shape[0] == BATCH_SIZE)
		assert(batch_y.shape[1] == 2)

def test_num_output_features(make_datagen):
	datagen = make_datagen
	x_batch, y_batch = next(datagen)
	assert(x_batch.shape[1] == datagen.num_features)