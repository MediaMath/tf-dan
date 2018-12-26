import sys
import pytest
import tensorflow as tf

sys.path.append('..')

from dan.disguise import Disguise
from dan.synthetic_data import make_two_gaussians

"""Configurable settings"""

# Dimensionality of the synthetic data. 2 allows for easy visualization
NUM_FEATURES = 2

# Number of samples in each class
NUM_SAMPLES = 1000
POSITIVE_PROP = 0.0

# Network architecture
HIDDEN_NODES = [32, 16, 8, 16, 32]

"""Fixtures"""

@pytest.fixture()
def make_data_and_model():
	model = Disguise(NUM_FEATURES, HIDDEN_NODES)
	X, y = make_two_gaussians(NUM_FEATURES, NUM_SAMPLES, POSITIVE_PROP)
	return model, X, y

"""Tests"""

def test_initialization():
	disguise = Disguise(NUM_FEATURES, HIDDEN_NODES, batch_norm=False)
	assert(len(disguise.layers) == len(HIDDEN_NODES) + 1)

def test_initialization_with_batch_norm():
	disguise = Disguise(NUM_FEATURES, HIDDEN_NODES, batch_norm=True)
	assert(len(disguise.layers) == len(HIDDEN_NODES) * 2 + 1)

def test_initizliation_with_batch_norm_dropout():
	disguise = Disguise(
		NUM_FEATURES, HIDDEN_NODES, batch_norm=True, dropout=0.5)
	assert(len(disguise.layers) == len(HIDDEN_NODES) * 3 + 1)

def test_transformation(make_data_and_model):
	disguise, X, y = make_data_and_model

	X_input = tf.constant(X, dtype=tf.float32)
	X_output = disguise(X_input)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		_ = sess.run(init)
		X_output_np = sess.run(X_output)

	assert(X_output_np.shape == X.shape)

