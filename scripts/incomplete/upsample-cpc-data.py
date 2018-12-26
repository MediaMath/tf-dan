import os

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, ADASYN

INTERMEDIATE_DIR = '../data/intermediate/mm-cpc/'
TRAIN_POSITIVE_NAME = 'train-positive.csv'

intermediate_dir_exists = os.path.isdir(INTERMEDIATE_DIR)
assert(intermediate_dir_exists),\
	'Intermediate data converted from raw do not exist.'

train_positive_path = os.path.join(INTERMEDIATE_DIR, TRAIN_POSITIVE_NAME)

df_positive = pd.read_csv(train_positive_path)
