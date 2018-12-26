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
from dan.data import load_jsons, load_csv_paths, DataGenerator

"""Configurable settings"""

# Dimensionality of the synthetic data. 2 allows for easy visualization
CHECKPOINT_DIR = 'checkpoints/'
LOG_DIR = 'logs/'

TRAIN_DIR = '../data/processed/mm-cpc-generator/train/'
VAL_DIR = '../data/processed/mm-cpc-generator/validation/'
TARGET = 'conversion_target'
BATCH_SIZE = 128
NUM_EPOCHS = 2

"""Fixtures"""

@pytest.fixture()
def make_datagen_and_model():
	tf.reset_default_graph()
	train_csv_paths = load_csv_paths(TRAIN_DIR, train=True)[:2]
	validation_csv_paths = load_csv_paths(TRAIN_DIR, train=True)[2:4]
	vocab_map, stats_map = load_jsons(TRAIN_DIR)
	train_datagen = DataGenerator(
		train_csv_paths, vocab_map, stats_map,
		batch_size=BATCH_SIZE,
		target='conversion_target',
		drop=['column_weights'],
		buffer_size=100000)
	validation_datagen = DataGenerator(
		validation_csv_paths, vocab_map, stats_map,
		batch_size=BATCH_SIZE,
		target='conversion_target',
		drop=['column_weights'],
		buffer_size=1)
	# disguise = Disguise(train_datagen.num_features, [128])
	disguise = None
	discriminator = Discriminator([]) # logistic regression
	model = DAN(
		disguise, discriminator, CHECKPOINT_DIR, LOG_DIR,
		num_inputs=train_datagen.num_features)
	return train_datagen, validation_datagen, model

"""Tests"""

def test_training(make_datagen_and_model):
	train_datagen, validation_datagen, dan = make_datagen_and_model
	dan.fit_generator(
		train_datagen,
		val_datagen=validation_datagen,
		num_epochs=NUM_EPOCHS)

def test_predict_generator(make_datagen_and_model):
	train_datagen, validation_datagen, dan = make_datagen_and_model
	dan.predict_generator(validation_datagen)

def test_predict_proba_generator(make_datagen_and_model):
	train_datagen, validation_datagen, dan = make_datagen_and_model
	dan.predict_proba_generator(validation_datagen)

