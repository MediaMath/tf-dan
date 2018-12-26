
import os
import glob
import pandas as pd
import argparse as ap

from functools import reduce
from tensorboard.backend.event_processing.event_accumulator \
  import EventAccumulator
  
parser = ap.ArgumentParser()
parser.add_argument(
  '-l', '--log_dir', type=str,
  help='Path to directory containing Tensorboard log files.')
parser.add_argument(
  '-c', '--csv_path', type=str,
  help='Path to output CSV file.')
args = parser.parse_args()

LOG_DIR = args.log_dir
CSV_PATH = args.csv_path

logs = glob.glob(os.path.join(LOG_DIR, '*'))
print('Found {} log files.'.format(len(logs)))
dfs = list()
for log in logs:
  event_accumulator = EventAccumulator(log)
  event_accumulator.Reload()
  event_accumulator.Tags()
  
  event_dfs = list()
  tags = event_accumulator.Tags()['scalars']
  print('Found {} scalars.'.format(len(tags)))
  for tag in tags:
    data = event_accumulator.Scalars(tag)
    data = pd.DataFrame(data)
    data = data.drop('wall_time', axis=1)
    tag = tag.lower().split('/')[-1]
    data.columns = ['step', tag]
    event_dfs.append(data)
    
  print('Merging scalar data...')
  df = reduce(
    lambda left, right: pd.merge(left, right, how='outer', on='step'),
    event_dfs)
  dfs.append(df)

print('Combining data from all log files...')
df_all = pd.concat(dfs, axis='rows')
df_all = df_all.set_index('step')
df_all = df_all.interpolate('linear')
df_all = df_all.fillna(method='backfill')
df_all.to_csv(CSV_PATH)