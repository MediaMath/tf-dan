import sys
import pytest
import os
import pytest
import numpy as np
import tensorflow as tf

sys.path.append('..')

from dan.embedding import EmbeddingTransformer
from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.dan import DAN
from dan.synthetic_data import make_mixed_data


"""Configurable settings"""

NUM_SAMPLES = 10000
NUM_CONTINUOUS_FEATURES = 4
CARDINALITIES = [200, 100, 80, 50]
CHECKPOINT_DIR = 'checkpoints/'
LOG_DIR = 'logs/'

num_inputs = NUM_CONTINUOUS_FEATURES + len(CARDINALITIES)

indices = np.arange(
	NUM_CONTINUOUS_FEATURES,
	NUM_CONTINUOUS_FEATURES + len(CARDINALITIES))

index_map = {index: cardinality for index, cardinality \
	in zip(indices, CARDINALITIES)}

"""Fixtures"""

@pytest.fixture()
def make_data():
	X, y = make_mixed_data(
		NUM_SAMPLES, NUM_CONTINUOUS_FEATURES, CARDINALITIES)
	return X, y

@pytest.fixture()
def make_model():
	tf.reset_default_graph()
	embedding_transformer = EmbeddingTransformer(index_map=index_map)
	disguise = Disguise(
		embedding_transformer.calc_num_outputs(num_inputs),
		[16, 8, 8, 16])
	discriminator = Discriminator([16, 8])

	dan = DAN(
		disguise, discriminator, CHECKPOINT_DIR, LOG_DIR,
		embedding_transformer=embedding_transformer,
		num_inputs=num_inputs,
	)
	return dan

@pytest.fixture()
def make_model_without_disguise():
	tf.reset_default_graph()
	embedding_transformer = EmbeddingTransformer(index_map=index_map)
	discriminator = Discriminator([16, 8])

	dan = DAN(
		None, discriminator, CHECKPOINT_DIR, LOG_DIR,
		embedding_transformer=embedding_transformer,
		num_inputs=num_inputs,
	)
	return dan

"""Tests"""
def test_training(make_data, make_model):
	X, y = make_data
	dan = make_model
	dan.fit(X, y, num_epochs=1)
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_training_without_disguise(
	make_data, make_model_without_disguise):
	X, y = make_data
	dan = make_model_without_disguise
	dan.fit(X, y, num_epochs=1)
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_predicting(make_data, make_model):
	X, y = make_data
	dan = make_model
	# dan.fit(X, y, num_epochs=1)
	y_pred = dan.predict(X)
	y_pred_proba = dan.predict_proba(X)
	num_samples = len(y)
	assert(y_pred.shape == (num_samples, 2))
	assert(y_pred_proba.shape == (num_samples, 2))
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_predicting_without_disguise(
	make_data, make_model_without_disguise):
	X, y = make_data
	dan = make_model_without_disguise
	# dan.fit(X, y, num_epochs=1)
	y_pred = dan.predict(X)
	y_pred_proba = dan.predict_proba(X)
	num_samples = len(y)
	assert(y_pred.shape == (num_samples, 2))
	assert(y_pred_proba.shape == (num_samples, 2))
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_transforming(make_data, make_model):
	X, y = make_data
	dan = make_model
	# dan.fit(X, y, num_epochs=1)
	X_disguised = dan.transform(X)
	num_outputs = dan.embedding_transformer.calc_num_outputs(num_inputs)
	assert(X_disguised.shape == (NUM_SAMPLES, num_outputs))
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)
