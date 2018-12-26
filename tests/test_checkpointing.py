import sys

sys.path.append('..')

from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.dan import DAN
from dan.synthetic_data import make_two_gaussians

"""Configurable settings"""

# Dimensionality of the synthetic data. 2 allows for easy visualization
NUM_FEATURES = 2
CHECKPOINT_DIR = 'checkpoints-2/'
LOG_DIR = 'logs-2/'

# Number of samples in each class
NUM_SAMPLES = 100000
POSITIVE_PROP = 0.1

X, y = make_two_gaussians()

disguise = Disguise(NUM_FEATURES, [4, 6, 4])
discriminator = Discriminator([4, 4])
model = DAN(disguise, discriminator, CHECKPOINT_DIR, LOG_DIR)

model.fit(X, y, num_epochs=10)