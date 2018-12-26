import os
import sys
import csv

sys.path.append('..')

import argparse as ap
import dan.data as ut

from dan.embedding import EmbeddingTransformer
from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.dan import DAN
from sklearn.metrics import precision_score, recall_score

parser = ap.ArgumentParser()
parser.add_argument(
	'--gpu', type=int, default=0,
	help='GPU ID number. Indexed from 0.')
parser.add_argument(
	'--experiment_dir', type=str,
	help='Directory where checkpoints, logs, and results are stored.')
parser.add_argument(
	'--data_dir', type=str,
	default='../data/processed/mm-cpc',
	help='Directory containing processed CSVs.')
parser.add_argument(
	'--train_csv', type=str,
	default='train-*.csv',
	help='Glob-friendly name of CSVs containing training data.')
parser.add_argument(
	'--validation_csv', type=str,
	default='validation.csv',
	help='Name of CSV containing validation data.')
parser.add_argument(
	'--test_csv', type=str,
	default='test.csv',
	help='Name of CSV containing test data.')
parser.add_argument(
	'--target', type=str,
	default='conversion_target',
	help='Name of the target feature to be predicted.')
parser.add_argument(
	'--json', type=str,
	default='index-cardinality.json',
	help='Name of JSON containing index to cardinality mappings.')
parser.add_argument(
	'--num_epochs', type=int, default=100,
	help='Number of training epochs.')
parser.add_argument(
	'--batch_size', type=int, default=128,
	help='Number of samples per batch.')
parser.add_argument(
	'--learning_rate', type=float, default=1e-5,
	help='Rate at which weights are updated by gradients.')
parser.add_argument(
	'--lambd', type=float, default=1.,
	help='Amount of regularization to apply to disguise loss.')
parser.add_argument(
	'--eta', type=float, default=0.25,
	help='Importance of predictive entropy in discriminator loss.')
parser.add_argument(
	'--disguise_architecture', type=int, nargs='+',
	default=[128, 128, 128, 128],
	help='Topology of disguise network.')
parser.add_argument(
	'--discriminator_architecture', type=int, nargs='+',
	default=[])
parser.add_argument(
	'--no_disguise', action='store_true',
	help='Trains a discriminator with no disguise network component.')
parser.add_argument(
	'--test', action='store_true',
	help='Loads model and evaluates it on test set.')
args = parser.parse_args()

GPU = str(args.gpu)
EXPERIMENT_DIR = args.experiment_dir
DATA_DIR = args.data_dir
TRAIN_CSV = args.train_csv
VALIDATION_CSV = args.validation_csv
TEST_CSV = args.test_csv
JSON_NAME = args.json
TARGET = args.target

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LAMBDA = args.lambd
ETA = args.eta
DISGUISE_ARCHITECTURE = args.disguise_architecture
DISCRMINATOR_ARCHITECTURE = args.discriminator_architecture
NO_DISGUISE = args.no_disguise
TEST = args.test

os.environ['CUDA_VISIBLE_DEVICES'] = GPU

# Load data
if not TEST:
	X_train, y_train = ut.load_data(
		os.path.join(DATA_DIR, TRAIN_CSV),
		target='conversion_target')
	print('Training data has shape: ', X_train.shape)
	print('Training labels has shape: ', y_train.shape)

	X_val, y_val = ut.load_data(
		os.path.join(DATA_DIR, VALIDATION_CSV),
		target='conversion_target')
	print('Validation data has shape: ', X_val.shape)
	print('Validation labels has shape: ', y_val.shape)

	assert(X_train.shape[1] == X_val.shape[1])
	num_features = X_train.shape[1]
else:
	X_test, y_test = ut.load_data(
	os.path.join(DATA_DIR, TEST_CSV),
	target='conversion_target')
	print('Test data has shape: ', X_test.shape)
	print('Test labels has shape: ', y_test.shape)
	num_features = X_test.shape[1]

# Create embedding map
json_path = os.path.join(DATA_DIR, JSON_NAME)
index_cardinality_map = ut.load_cardinality_map(json_path)

# Create new directories / define directory paths
if not os.path.isdir(EXPERIMENT_DIR):
	os.makedirs(EXPERIMENT_DIR)

checkpoint_dir = os.path.join(EXPERIMENT_DIR, 'checkpoints/')
log_dir = os.path.join(EXPERIMENT_DIR, 'logs/')

# Define DAN model
embedding_transformer = EmbeddingTransformer(index_cardinality_map)
num_features_embedding = embedding_transformer.calc_num_outputs(num_features)
print('Feature dimensionality in embedded space:', num_features_embedding)

if NO_DISGUISE or TEST:
	disguise = None
else:
	disguise = Disguise(num_features_embedding, DISGUISE_ARCHITECTURE)

discriminator = Discriminator(DISCRMINATOR_ARCHITECTURE)
dan = DAN(
	disguise, discriminator, checkpoint_dir, log_dir, embedding_transformer,
	num_inputs=num_features,
	learning_rate=LEARNING_RATE,
	lambd=LAMBDA,
	eta=ETA,
	early_stopping=4)

if not TEST:
	print('Logging hyperparmeters...')
	log_path = os.path.join(EXPERIMENT_DIR, 'hyperparameters.txt')
	with open(log_path, 'at') as f:
		lines = list()
		lines.append('num_epochs: {}\n'.format(NUM_EPOCHS))
		lines.append('batch_size: {}\n'.format(BATCH_SIZE))
		lines.append('learning_rate: {}\n'.format(LEARNING_RATE))
		lines.append('lambda: {}\n'.format(LAMBDA))
		lines.append('eta: {}\n'.format(ETA))
		if NO_DISGUISE:
			lines.append('no_disguise: {}\n'.format(NO_DISGUISE))
		else:
			lines.append(
				'disguise_architecture: {}\n'.format(DISGUISE_ARCHITECTURE))
		lines.append(
			'discriminator_architecture: {}\n'.format(DISCRMINATOR_ARCHITECTURE))
		lines.append('\n')
		f.writelines(lines)

	print('Training DAN...')
	dan.fit(
		X_train, y_train,
		num_epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		x_val=X_val, y_val=y_val)

else:
	print('Evaluating DAN...')
	metrics = dan.evaluate(X_test, y_test)
	cross_entropy, accuracy, auroc = metrics

	y_pred = dan.predict(X_test)
	y_pred = y_pred[:,1]
	y_test = y_test[:,1]
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	best_val_auroc = dan._best_val_auroc

	csv_path = os.path.join(EXPERIMENT_DIR, 'test_results.csv')
	print('Writing results to {}'.format(csv_path))
	with open(csv_path, 'w') as f:
		writer = csv.writer(f)
		writer.writerow([
			'cross_entropy',
			'accuracy',
			'auroc',
			'precision',
			'recall',
			'val_auroc'])
		writer.writerow([
			cross_entropy, 
			accuracy, 
			auroc,
			precision,
			recall,
			best_val_auroc])