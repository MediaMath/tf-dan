import sys
import pytest
import os
import tensorflow as tf

sys.path.append('..')

from dan.embedding import EmbeddingTransformer
from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.dan import DAN
from dan.synthetic_data import make_two_gaussians

"""Configurable settings"""

# Dimensionality of the synthetic data. 2 allows for easy visualization
NUM_FEATURES = 2
CHECKPOINT_DIR = 'checkpoints/'
LOG_DIR = 'logs/'

# Number of samples in each class
NUM_SAMPLES = 100000
POSITIVE_PROP = 0.1

"""Fixtures"""

@pytest.fixture()
def make_data():
	X, y = make_two_gaussians(NUM_FEATURES, NUM_SAMPLES, POSITIVE_PROP)
	return X, y

@pytest.fixture()
def make_model():
	tf.reset_default_graph()
	disguise = Disguise(NUM_FEATURES, [4, 6, 4])
	discriminator = Discriminator([4, 4])
	model = DAN(disguise, discriminator, CHECKPOINT_DIR, LOG_DIR)
	return model

@pytest.fixture()
def make_model_without_disguise():
	tf.reset_default_graph()
	discriminator = Discriminator([4, 4])
	model = DAN(
		None, discriminator, CHECKPOINT_DIR, LOG_DIR, num_inputs=NUM_FEATURES)
	return model

"""Tests"""

def test_initialization():
	disguise = Disguise(NUM_FEATURES, [4, 6, 4])
	discriminator = Discriminator([4, 4])
	dan = DAN(disguise, discriminator, CHECKPOINT_DIR, LOG_DIR)

def test_initialization_without_disguise():
	discriminator = Discriminator([4, 4])
	dan = DAN(None, discriminator, CHECKPOINT_DIR, LOG_DIR, num_inputs=2)

def test_initialization_with_error():
	try:
		discriminator = Discriminator([4, 4])
		dan = DAN(None, discriminator, CHECKPOINT_DIR, LOG_DIR)
	except ValueError:
		pass

def test_training(make_data, make_model):
	X, y = make_data
	dan = make_model
	dan.fit(X, y, num_epochs=1)
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_predicting(make_data, make_model):
	X, y = make_data
	dan = make_model
	y_pred = dan.predict(X)
	y_pred_proba = dan.predict_proba(X)
	num_samples = len(y)
	assert(y_pred.shape == (num_samples, NUM_FEATURES))
	assert(y_pred_proba.shape == (num_samples, NUM_FEATURES))

def test_transforming(make_data, make_model):
	X, y = make_data
	dan = make_model
	X_disguised = dan.transform(X)
	assert(X_disguised.shape == X.shape)

def test_training_without_disguise(
	make_data, make_model_without_disguise):

	X, y = make_data
	dan = make_model_without_disguise
	dan.fit(X, y, num_epochs=1)
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_predicting_without_disguise(
	make_data, make_model_without_disguise):

	X, y = make_data
	dan = make_model_without_disguise
	y_pred = dan.predict(X)
	y_pred_proba = dan.predict_proba(X)
	num_samples = len(y)
	assert(y_pred.shape == (num_samples, NUM_FEATURES))
	assert(y_pred_proba.shape == (num_samples, NUM_FEATURES))
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)

def test_transforming_without_disguise(
	make_data, make_model_without_disguise):

	X, y = make_data
	dan = make_model_without_disguise
	try:
		X_disguised = dan.transform(X)
	except AttributeError:
		pass
	os.system('rm -r '+CHECKPOINT_DIR)
	os.system('rm -r '+LOG_DIR)