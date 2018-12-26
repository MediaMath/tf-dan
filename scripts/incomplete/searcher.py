import os
import numpy as np
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument(
	'--lower_lambd', type=int, default=0,
	help='lowest value of lambda to try')
parser.add_argument(
	'--upper_lambd', type=int, default=2,
	help='highest value of lambda to try')
parser.add_argument(
	'--number_of_lambd', type=int, default=10,
	help='number of lambda values to try')
parser.add_argument(
	'--eta', type=int, default=0,
	help="it's a constant for now")

args = parser.parse_args()

i = 0
lambd = np.linspace(args.lower_lambd,args.upper_lambd, args.number_of_lambd)
eta = [args.eta]*101

while True:
    os.system('export a=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)')
    free_mem = np.array([int(i) for i in os.environ['a'].split(',')])
    all_free = free_mem>7000
    if sum(all_free)>0 :
        first_free = np.argmax(all_free)
        os.system(f"screen -dmS experiment-{i} bash -c 'python3.6 run-experiment-cpc.py --experiment_dir home/f18-mediamath/models/experiment-{i} --gpu {first_free} --lambd {lambd[i]} --eta {eta[i]}'")
        i+=1
    if i>= len(lambd):
        break
