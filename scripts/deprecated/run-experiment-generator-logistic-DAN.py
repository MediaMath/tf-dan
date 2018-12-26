import os
import sys
import csv

sys.path.append('..')

import argparse as ap

from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.dan import DAN
from dan.data import load_jsons, load_csv_paths, DataGenerator
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
tf.reset_default_graph()
parser = ap.ArgumentParser()
parser.add_argument(
	'--gpu', type=int, default=0,
	help='GPU ID number. Indexed from 0.')
parser.add_argument(
	'--experiment_dir', type=str,
	help='Directory where checkpoints, logs, and results are stored.')
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
	'--buffer_size', type=int, default=70000,
	help='Number of look-ahead samples for pseudo-random shuffling.')
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
	default=[256, 128, 128, 256],
	help='Topology of disguise network.')
parser.add_argument(
	'--discriminator_architecture', type=int, nargs='*',
	default=[])
parser.add_argument(
	'--no_disguise', action='store_true',
	help='Trains a discriminator with no disguise network component.')
parser.add_argument(
	'--test', action='store_true',
	help='Loads model and evaluates it on test set.')
args = parser.parse_args()

GPU = '0'#str(args.gpu)
EXPERIMENT_DIR = args.experiment_dir
DATASET = args.dataset
TARGET = args.target
DROP = ['column_weights']

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
BUFFER_SIZE = args.buffer_size
LEARNING_RATE = args.learning_rate
LAMBDA = args.lambd
ETA = args.eta
DISGUISE_ARCHITECTURE = []#args.disguise_architecture
DISCRIMINATOR_ARCHITECTURE = []#args.discriminator_architecture
NO_DISGUISE = True#args.no_disguise
TEST = True#args.test

os.environ['CUDA_VISIBLE_DEVICES'] = GPU

if DATASET == 'mm-cpc':
	data_dir = '../data/processed/mm-cpc-generator/'
elif DATASET == 'mm-ctr':
	data_dir = '../data/processed/mm-ctr-generator/'
print('Using dataset at {}'.format(data_dir))
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

# Load data
vocab_map, stats_map = load_jsons(train_dir)

print('Loading training data generator...')
train_csv_paths = load_csv_paths(train_dir, train=True)

neg_paths = []
pos_paths = []
ct = 0 

#selecting 12 shards of negative and all positive shards

for path in train_csv_paths:
    if 'negative' in path and ct <12:
        neg_paths.append(path)
        ct += 1
    elif 'positive' in path:
        pos_paths.append(path)

#follwing for loop distributes positve and negative sharded csv evenly
#this loop distribution works only in case of 100 shards
train_csv_paths = []
for i in range(len(pos_paths)):
    if i!=0 and i%8 == 0:
        train_csv_paths.append(pos_paths[i])
        train_csv_paths.append(neg_paths[int(i/8)-1])
    else:
        train_csv_paths.append(pos_paths[i])
        
#print  ("'%04d'"%12)
#
#train_csv_paths = ['../data/processed/mm-cpc-generator/train/train-negative-0000.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0000.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0001.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0002.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0003.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0004.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0005.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0006.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0007.csv',
# '../data/processed/mm-cpc-generator/train/train-positive-0008.csv']

train_datagen = DataGenerator(
	train_csv_paths, vocab_map, stats_map,
	batch_size=BATCH_SIZE,
	target=TARGET,
	drop=DROP,
	buffer_size=BUFFER_SIZE)
print('Training data has {} samples.'.format(
	train_datagen.num_samples))
neg_labels_pct = 34330*ct*100/train_datagen.num_samples
print('Out of which +ve class labels:{:.1f}% and -ve \
      class labels:{:.1f}%'.format(100-neg_labels_pct,neg_labels_pct))
print('Training data has {} features.'.format(
	train_datagen.num_features))

print('Loading validation data generator...')
validation_csv_paths = load_csv_paths(validation_dir, train=True)
validation_datagen = DataGenerator(
	validation_csv_paths, vocab_map, stats_map,
	batch_size=BATCH_SIZE,
	target=TARGET,
	drop=DROP,
	buffer_size=1) # No random shuffling
print('Validation data has {} samples.'.format(
	validation_datagen.num_samples))
print('Validation data has {} features.'.format(
	validation_datagen.num_features))
num_features = train_datagen.num_features

print('Loading test data generator...')
test_csv_paths = load_csv_paths(test_dir, train=False)
test_datagen = DataGenerator(
	test_csv_paths, vocab_map, stats_map,
	batch_size=BATCH_SIZE,
	target=TARGET,
	drop=DROP,
	buffer_size=1) # No random shuffling
print('Test data has {} samples.'.format(
	test_datagen.num_samples))
print('Test data has {} features.'.format(
	test_datagen.num_features))
num_features = test_datagen.num_features

# Create new directories / define directory paths
if not os.path.isdir(EXPERIMENT_DIR):
	os.makedirs(EXPERIMENT_DIR)

checkpoint_dir = os.path.join(EXPERIMENT_DIR, 'checkpoints/')
log_dir = os.path.join(EXPERIMENT_DIR, 'logs/')

# Define DAN model
print('Building DAN...')
if NO_DISGUISE or TEST:
	disguise = None
else:
	disguise = Disguise(train_datagen.num_features, DISGUISE_ARCHITECTURE)
discriminator = Discriminator(DISCRIMINATOR_ARCHITECTURE)
dan = DAN(
	disguise, discriminator, checkpoint_dir, log_dir,
	num_inputs=num_features,
	learning_rate=LEARNING_RATE,
	lambd=LAMBDA,
	eta=ETA)


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
		'discriminator_architecture: {}\n'.format(DISCRIMINATOR_ARCHITECTURE))
	lines.append('\n')
	f.writelines(lines)

print('Training DAN...')
dan.fit_generator(
	train_datagen,
	num_epochs=NUM_EPOCHS,
	val_datagen=validation_datagen)

print('Evaluating DAN...')
metrics = dan.evaluate_generator(test_datagen)
cross_entropy, accuracy, auroc = metrics

y_pred, y_test = dan.predict_generator(test_datagen)
y_pred = y_pred[:,1]
y_test = y_test[:,1]
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

csv_path = os.path.join(EXPERIMENT_DIR, 'test_results.csv')
print('Writing results to {}'.format(csv_path))
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