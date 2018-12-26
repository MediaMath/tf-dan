"""
Using the Test Tube API to log multiple experiments
Change Log: 
1. Removed TEST argument and associated functionalities (since we always test)
2. Removed NO_DISGUISE argument (always disguise)
3. Removed DROP 
4. Changed EXPERIMENT_DIR to TESTTUBE_DIR 
5. Dynamically Named EXPERIMENT_DIR 
6. TODO Statically set disguise and discriminator structure due to test_tube bug/limitation
		- self.__parse_args(args, namespace)
		- __whitelist_cluster_commands
		- all_values.add(v) #error here: this tries to hash values passed in, but can't hash a list 
		- TypeError: unhashable type: 'list'
		- current fix: convert to tuple and then list 
7. TODO: Makes sure: '../data/processed/mm-cpc-generator/train/categorical-vocab.json' exists 
7.1. Process for Generator first 
8. Moved testtube_dir into the 'models' folder
99. TODO: README 

"""

import os
import sys
import csv
import glob
import time 

sys.path.append('..')

import argparse as ap
from test_tube import Experiment, HyperOptArgumentParser

import tensorflow as tf 
from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.dan import DAN
from dan.data import load_jsons, load_csv_paths, DataGenerator

from sklearn.metrics import precision_score, recall_score


"""
-------------------------------------------------------------------------------------------------------------------------------------------------
Part 1: Define Module
-------------------------------------------------------------------------------------------------------------------------------------------------
""" 
#EXPERIMENT_DIR, DATASET, TARGET, NUM_EPOCHS, BATCH_SIZE, BUFFER_SIZE, LEARNING_RATE, LAMBDA, ETA, DISGUISE_ARCHITECTURE, DISCRIMINATOR_ARCHITECTURE
def TestTubeGenerator(hparams): 

	"""
	Trains, Validates, and Tests specific dataset while iterating through ETA and LAMBDA hyperparameters
	>>> Command Line Usage: (*) indicates minimally required : 
	python run-testtube-generator.py 
		--dataset (*)
		--target 
		--num_epochs
		--batch_size
		--buffer_size
		--learning_rate
		--lambd
		--eta
		--disguise_architecture
		--discriminator_architecture 
	"""
	# TESTTUBE_DIR = hparams.testtube_dir #changed to TESTTUBE DIR 
	DATASET = hparams.dataset
	TARGET = hparams.target
	NUM_EPOCHS = hparams.num_epochs
	BATCH_SIZE = hparams.batch_size
	BUFFER_SIZE = hparams.buffer_size
	LEARNING_RATE = hparams.learning_rate
	LAMBDA = hparams.lambd
	ETA = hparams.eta
	DISGUISE_ARCHITECTURE = list(hparams.disguise_architecture)
	# DISCRIMINATOR_ARCHITECTURE = list(hparams.discriminator_architecture)
	DISCRIMINATOR_ARCHITECTURE = [] 

	"""-----------------------------Section 1: Directories and Paths-----------------------------""" 
	#1.1 Setting Directory for Dataset (train, validation, test)
	if DATASET == 'mm-cpc':
		data_dir = '../data/processed/mm-cpc-generator/'
		model_dir = '../models/mm-cpc-onehot/'
	elif DATASET == 'mm-ctr':
		data_dir = '../data/processed/mm-ctr-generator/'
		model_dir = '../models/mm-ctr-onehot/'

	train_dir = os.path.join(data_dir, 'train')
	validation_dir = os.path.join(data_dir, 'validation')
	test_dir = os.path.join(data_dir, 'test')
	print('Using dataset at {}'.format(data_dir))

	#1.1.1 make a testtube dir if it doesn't exist yet 
	testtube_dir = model_dir + 'testtube_test_logistic_1_small/'
	if not os.path.exists(testtube_dir): 
		os.makedirs(testtube_dir)

	#1.1.2 make experiment directory withing testtube dir 
	current_time = str(time.time())
	experiment_dir = testtube_dir+ "experiment" + "_" + current_time + "_"+str(LAMBDA)+"_"+str(ETA)
	assert(not os.path.exists(experiment_dir))
	os.makedirs(experiment_dir)

	#1.1.3 set experiment 
	exp = Experiment(name='DAN'+"_"+str(LAMBDA)+"_"+str(ETA), debug=False, save_dir=experiment_dir)
	exp.argparse(hparams)

	#1.2 load the categorical and numerical maps associated with the Data
	vocab_map, stats_map = load_jsons(train_dir)

	checkpoint_dir = os.path.join(experiment_dir, 'checkpoints/')
	log_dir = os.path.join(experiment_dir, 'logs/')
	print('Experiment Directory Defined at {}'.format(experiment_dir))

	print("-"*50)
	# print(load_csv_paths(train_dir, train=True))

	"""-----------------------------Section 2: Training and Validation-----------------------------"""
	#2.1 Training and Validation Data Generators and Summary
	tf.reset_default_graph() #TODO make sure this takes care of the thread issue

	train_csv_paths = load_csv_paths(train_dir, train=True)[:2]
	train_datagen = DataGenerator(
		train_csv_paths, vocab_map, stats_map,
		batch_size=BATCH_SIZE,
		target=TARGET,
		drop=['column_weights'],
		buffer_size=BUFFER_SIZE)
	print('Training data has {} samples.'.format(train_datagen.num_samples))
	print('Training data has {} features.'.format(train_datagen.num_features))

	print('Loading validation data generator...')
	validation_csv_paths = load_csv_paths(train_dir, train=True)[2:4]
	validation_datagen = DataGenerator(
		validation_csv_paths, vocab_map, stats_map,
		batch_size=BATCH_SIZE,
		target=TARGET,
		drop=['column_weights'],
		buffer_size=1) # No random shuffling
	print('Validation data has {} samples.'.format(validation_datagen.num_samples))
	print('Validation data has {} features.'.format(validation_datagen.num_features))

	#2.2 Define Disguise
	disguise = Disguise(train_datagen.num_features, DISGUISE_ARCHITECTURE)
	print('Disguise Defined with {} architecture.'.format(DISGUISE_ARCHITECTURE))

	#2.3 Define Discriminator 
	discriminator = Discriminator(DISCRIMINATOR_ARCHITECTURE)
	dan = DAN(
		disguise, discriminator, checkpoint_dir, log_dir,
		num_inputs=train_datagen.num_features,
		learning_rate=LEARNING_RATE,
		lambd=LAMBDA,
		eta=ETA)
	print('Discriminator Defined with {} architecture.'.format(DISCRIMINATOR_ARCHITECTURE))

	#2.4 Training Disguise Adversarial Network 
	print('Training DAN.')
	dan.fit_generator(
		train_datagen,
		num_epochs=NUM_EPOCHS,
		val_datagen=validation_datagen)
	print('Training Complete.')

	#2.5.1 Logging Parameters (epochs, batch, learning rate, lambda, eta)
	best_auroc = dan._best_val_auroc

	log_path = os.path.join(experiment_dir, 'hyperparameters.txt')
	with open(log_path, 'at') as f:
		lines = list()
		lines.append('num_epochs: {}\n'.format(NUM_EPOCHS))
		lines.append('batch_size: {}\n'.format(BATCH_SIZE))
		lines.append('learning_rate: {}\n'.format(LEARNING_RATE))
		lines.append('lambda: {}\n'.format(LAMBDA))
		lines.append('eta: {}\n'.format(ETA))
		lines.append('discriminator_architecture: {}\n'.format(DISCRIMINATOR_ARCHITECTURE))
		lines.append('best auroc: {}\n'.format(best_auroc))
		lines.append('\n')
		f.writelines(lines)
	print("Parameters Logged at {}.".format(log_path))

	#2.5.2 Tagging Parameters with EXPERIMENT API
	exp.tag({
		'num_epochs': NUM_EPOCHS, 
		'batch_size': BATCH_SIZE, 
		'learning_rate': LEARNING_RATE, 
		'lambda': LAMBDA, 
		'eta': ETA, 
		'discriminator_architecture': DISCRIMINATOR_ARCHITECTURE, 
		'best_auroc': best_auroc})
	
	print("Experiment Tagged.")

	"""----------------------------------Section 3: Testing----------------------------------"""
	print("Initializing Testing Phase.")

	#3.1 Test Data Generator and Summary 
	test_csv_paths = load_csv_paths(train_dir, train=False)[:2]
	test_datagen = DataGenerator(
		test_csv_paths, vocab_map, stats_map,
		batch_size=BATCH_SIZE,
		target=TARGET,
		drop=['column_weights'],
		buffer_size=1) # No random shuffling
	print('Test data has {} samples.'.format(test_datagen.num_samples))
	print('Test data has {} features.'.format(test_datagen.num_features))

	#3.2.1 Evaluation: CE, ACC, AUROC
	print("Evaluating.")
	metrics = dan.evaluate_generator(test_datagen)
	cross_entropy, accuracy, auroc = metrics
	print('The Cross Entropy, Accuracy, and AUROC are: ', metrics)

	#3.2.2 Evaluation: Precision and Precall
	y_pred, y_test = dan.predict_generator(test_datagen)
	y_pred = y_pred[:,1]
	y_test = y_test[:,1]
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	print("The Precision and Recall are: ", precision, " and ", recall)

	#3.3.1 Result Logging: CSV log
	csv_path = os.path.join(experiment_dir, 'test_results.csv') #result logging directory 
	with open(csv_path, 'w') as f:
		writer = csv.writer(f)
		writer.writerow([
			'cross_entropy',
			'accuracy',
			'auroc',
			'precision',
			'recall'])
		writer.writerow([
			cross_entropy, 
			accuracy, 
			auroc,
			precision,
			recall])
		
	print('Results written to {}'.format(csv_path))

	#3.3.2 Result Logging with Experiment API 
	exp.log({
		'cross_entropy': cross_entropy, 
		'accuracy': accuracy, 
		'auroc': auroc, 
		'precison': precision, 
		'recall': recall})

	print("Experiment Logged.")

	exp.save()

"""
-------------------------------------------------------------------------------------------------------------------------------------------------
Part II: Parse and Run 
-------------------------------------------------------------------------------------------------------------------------------------------------
""" 
parser = HyperOptArgumentParser(strategy='random_search')

#set default values for directories 
# parser.add_argument(
# 	'--testtube_dir', type=str,
# 	help='Directory containing experiment folders.') #TODO set default 
parser.add_argument(
	'--dataset', type=str, required=True,
	help='Dataset to use. One of {mm-cpc, mm-ctr}.')
parser.add_argument(
	'--target', type=str,
	default='conversion_target',
	help='Name of the target feature to be predicted.')
parser.add_argument(
	'--num_epochs', type=int, default=100,
	help='Number of training epochs.')
parser.add_argument(
	'--batch_size', type=int, default=128,
	help='Number of samples per batch.')
parser.add_argument(
	'--buffer_size', type=int, default=100000,
	help='Number of look-ahead samples for pseudo-random shuffling.')
parser.add_argument(
	'--learning_rate', type=float, default=1e-5,
	help='Rate at which weights are updated by gradients.')
parser.opt_range(
	'--lambd', type=float, default=0.05, tunable=True, low=0.05, high=2.0, nb_samples=16, log_base=2,   
	help='Amount of regularization to apply to disguise loss.')
parser.opt_range(
	'--eta', type=float, default=0.05, tunable=True, low=0.05, high=2.0, nb_samples=8, log_base=2, 
	help='Importance of predictive entropy in discriminator loss.')
parser.add_argument(
	'--disguise_architecture', type=int, nargs='+',
	default=tuple([64, 32, 32, 64]),
	help='Topology of disguise network.')
# parser.add_argument(
# 	'--discriminator_architecture', type=int, nargs='*',
# 	default=tuple([64, 32, 16]))
parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])

hparams = parser.parse_args()

"""Run Trials""" 
#duplicate for more trials 
hparams.optimize_parallel_gpu(TestTubeGenerator, gpu_ids = ['0','1','2','3','4','5','6','7'], nb_trials = 8, nb_workers = 8)
