import os
import sys
import glob
import gzip
import shutil

from column_file_generator_json import column_file_mass_generator

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument(
	'--raw_dir', type=str, required=True,
	help=(
		'Directory containing folders for raw train, validation '
		'and test data.'))
parser.add_argument(
	'--output_dir', type=str, required=True,
	help=(
		'Directory where folders for processed train, validation '
		'and test data are stored.'))
parser.add_argument(
	'--num_shards', type=int, default=100,
	help='Number of shards in which to split the training data.')
args = parser.parse_args()

RAW_DIR = args.raw_dir
OUTPUT_DIR = args.output_dir
NUM_SHARDS = args.num_shards

if not os.path.isdir(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)
	for split in ['train', 'validation', 'test']:
		os.mkdir(os.path.join(OUTPUT_DIR, split))

train_gzip_paths = glob.glob(os.path.join(RAW_DIR, 'train', '*', '*.gz'))
other_gzip_paths = glob.glob(os.path.join(RAW_DIR, '*', '*.gz'))
gzip_paths = train_gzip_paths + other_gzip_paths
csv_paths = [gzip_path.rstrip('.gz') for gzip_path in gzip_paths]

for gzip_path, csv_path in zip(gzip_paths, csv_paths):
	print('Uncompressing GZ file: {}'.format(gzip_path))
	with gzip.open(gzip_path, 'rb') as f_in:
		with open(csv_path, 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)

print('Copying test and validation CSVs to new directories...')
for csv_path in csv_paths:
	if 'validation' in os.path.dirname(csv_path):
		shutil.copy2(csv_path, os.path.join(OUTPUT_DIR, 'validation'))
	elif 'test' in os.path.dirname(csv_path):
		shutil.copy2(csv_path, os.path.join(OUTPUT_DIR, 'test'))

def merge_csv(csv_paths, output_path):
	with open(output_path, 'wt') as f_out:
		header = False
		for csv_path in csv_paths:
			with open(csv_path, 'rt') as f_in:
				if header is True:
					f_in.readline()
				else:
					header = True
				for line in f_in:
					f_out.write(line)

print('Combining train CSVs...')
pos_csv_paths = [csv_path \
	for csv_path \
	in csv_paths \
	if 'data-positive' in csv_path]
temp_csv_path = os.path.join(
	RAW_DIR, 'train', 'train-positive.csv')
merge_csv(pos_csv_paths, temp_csv_path)

neg_csv_paths = [csv_path \
	for csv_path \
	in csv_paths \
	if 'data-negative' in csv_path]
temp_csv_path = os.path.join(
	RAW_DIR, 'train', 'train-negative.csv')
merge_csv(neg_csv_paths, temp_csv_path)

print('Sharding train CSVs...')
command = [
	'python shard-data.py',
	'--source_dir {}'.format(os.path.join(RAW_DIR, 'train')),
	'--destination_dir {}'.format(os.path.join(OUTPUT_DIR, 'train')),
	'--source_files {} {}'.format(
		'train-positive.csv', 'train-negative.csv'),
	'--num_shards {}'.format(NUM_SHARDS)]
command = ' '.join(command)
os.system(command)

print('Generating vocabulary and stats JSONs...')
json_output_dir = os.path.join(OUTPUT_DIR, 'train/')
column_file_mass_generator(json_output_dir, json_output_dir)

print('Cleaning up...')
for csv_path in csv_paths:
	os.remove(csv_path)
for csv_path in glob.glob(os.path.join(RAW_DIR, 'train', '*.csv')):
	os.remove(csv_path)
