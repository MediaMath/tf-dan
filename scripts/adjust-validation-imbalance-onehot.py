import os
import glob
import random

import numpy as np
import pandas as pd

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument(
	'--dataset', type=str, required=True,
	help=('Dataset to use: mm-cpc'))
parser.add_argument(
	'--pos_prop', type=int, default=11,
	help=(
		'proportion of pos samples required in val data'))

args = parser.parse_args()


#RAW_DIR = args.raw_dir
DATASET = args.dataset#'mm-cpc'
POSITIVE_SAMPLE_PROPORTION = args.pos_prop#11
if DATASET == 'mm-cpc':
    val_dir = '../data/processed/mm-cpc-generator/validation/'
val_csv = glob.glob(os.path.join(val_dir, '*.csv'))

val_df = pd.read_csv(val_csv[-1],index_col=None, header=0)
print ('initial shape of val_data:',val_df.shape)
#divides + ve and -ve indices from df
pos_idx = np.where(val_df.conversion_target == 1)[0]
neg_idx = np.where(val_df.conversion_target == 0)[0]

print('out of which 1s are:{} 0s are:{}'.format(len(pos_idx),len(neg_idx)))


#number of negative samples needed from validation dataset is:
pos_samples_ct_reqd = round(POSITIVE_SAMPLE_PROPORTION*len(neg_idx)/(100-POSITIVE_SAMPLE_PROPORTION))

#undersampling +ve labels to match proportion
pos_idx_df_selected = random.sample(list(pos_idx), pos_samples_ct_reqd)

#combining neg and pos df indices
idx_selected = pos_idx_df_selected + list(neg_idx)
idx_selected = random.sample(idx_selected, len(idx_selected))

print('after adjusting, 1s: {} and 0s: {}'.format(len(pos_idx_df_selected),len(neg_idx)))

#final df building
val_df_final = val_df.iloc[idx_selected,:].reset_index(drop=True).copy()

#overwriting existing csv file in val set
val_df_final.to_csv(val_csv[-1],index = False)