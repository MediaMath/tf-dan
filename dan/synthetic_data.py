import numpy as np
import pandas as pd

def make_two_gaussians(
	num_features=2, num_samples=100000,
	positive_prop=0.1, negative_mu=0, negative_sigma=1, positive_mu=1,
	positive_sigma=0.4):

	num_positives = int(round(num_samples * positive_prop))
	num_negatives = num_samples - num_positives

	# Make positive/negative samples and combine
	X_negatives = np.random.normal(
	  loc=[negative_mu]*num_features,
	  scale=[negative_sigma]*num_features, 
	  size=[num_negatives, num_features])
	X_positives = np.random.normal(
	  loc=[positive_mu]*num_features,
	  scale=[positive_sigma]*num_features,
	  size=[num_positives, num_features])
	X = np.vstack([X_negatives, X_positives])

	# Generate labels and one-hot encode them
	y = np.array([0] * num_negatives + [1] * num_positives)
	y = pd.get_dummies(y).values

	# Randomly shuffle samples and labels
	shuffled_indices = np.random.permutation(range(len(X)))
	X = X[shuffled_indices]
	y = y[shuffled_indices]

	return X, y

def make_mixed_data(
	num_samples=100000, num_continuous_features=4,
	cardinalities=[10, 100, 200, 50]):
	
	X_continuous = np.random.randn(num_samples, num_continuous_features)
	categorical_features = list()
	for cardinality in cardinalities:
		values = np.arange(cardinality)
		feature = np.random.choice(values, [num_samples, 1])
		categorical_features.append(feature)
	X_categorical = np.column_stack(categorical_features)
	X = np.column_stack([X_continuous, X_categorical])
	y = np.random.choice([0, 1], num_samples)
	y = pd.get_dummies(y).values
	return X, y
