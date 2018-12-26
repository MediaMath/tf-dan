import os
import csv

import itertools as it
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument(
	'--source_dir', type=str,
	default='../data/intermediate/mm-cpc/',
	help='Directory of intermediate CSV\'s to be sharded.')
parser.add_argument(
	'--destination_dir', type=str,
	default='../data/processed/mm-cpc/',
	help='Directory of sharded CSV\'s.')
parser.add_argument(
	'--source_files', type=str, nargs='+', required=True,
	help='Names of CSV\'s to be sharded.')
parser.add_argument(
	'--num_shards', type=int, default=100,
	help='Number of shards.')
args = parser.parse_args()

SOURCE_DIR = args.source_dir
DESTINATION_DIR = args.destination_dir
SOURCE_FILES = args.source_files
NUM_SHARDS = args.num_shards

if not os.path.isdir(DESTINATION_DIR):
	os.makedirs(DESTINATION_DIR)

for source_file in SOURCE_FILES:
	source_path = os.path.join(SOURCE_DIR, source_file)
	source_file_list = source_file.split('.')
	source_file_list.insert(1, '-{:0>4}.')
	dest_filename_format = ''.join(source_file_list)
	dest_path_format = os.path.join(DESTINATION_DIR, dest_filename_format)

	with open(source_path, 'rt') as f:
		num_samples = sum(1 for row in f) - 1
		num_samples_per_shard = num_samples // NUM_SHARDS
		print(f'{num_samples_per_shard} samples per shard for {source_file}.')

		remove_glob = os.path.join(
			DESTINATION_DIR, source_file_list[0]) + '-*.csv'
		os.system(f'rm {remove_glob}')

	with open(source_path, 'rt') as f_in:
		header = f_in.readline()
		for shard_counter in range(NUM_SHARDS + 1):
			with open(dest_path_format.format(shard_counter), 'wt') as f_out:
				f_out.write(header)
				for i in range(num_samples_per_shard):
					try:
						f_out.write(f_in.readline())
					except:
						pass