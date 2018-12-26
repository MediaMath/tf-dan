import numpy as np
import tensorflow as tf

class EmbeddingTransformer:
	
	def __init__(self, index_map, alpha=3):
		"""
		A callable that replaces the categorical features in input data with
		embedded representations. Non-categorical features remain unaltered.

		Parameters
		==========
		index_map : dict
			A dictionary where each key is the column index of a categorical
			feature and each value is the cardinality of that feature.

		Usage
		=====
		>>> emb_transformer = EmbeddingTransformer(
					index_map={3: 20, 8: 1000, 13: 45},
					alpha=4)

		Attributes
		==========
		embedding_map : dict
			A dictionary where each key is the column index of a categorical
			feature and each value is that feature's corresponding embedding
			matrix of shape (cardinality, embedding_dimension).
		"""
		self.embedding_map = dict()
		self.alpha = alpha
		self.total_embedding_dimensions = 0
		with tf.variable_scope('dan/embeddings', reuse=tf.AUTO_REUSE):
			for index, cardinality in index_map.items():
				embedding_dim = cardinality ** (1 / self.alpha)
				embedding_dim = int(np.ceil(embedding_dim))
				embedding_matrix = tf.get_variable(
					name=f'embeddings_{index}',
					shape=(cardinality, embedding_dim),
					dtype=tf.float32,
				)
				self.embedding_map[index] = embedding_matrix
				self.total_embedding_dimensions += embedding_dim

	def __call__(self, x):
		"""
		Replaces the categorical features in input data with embedded
		representations while non-categorical features remain unaltered.

		Parameters
		==========
		x : tensor
			A tensor or placeholder for the input data which may contain a mix of numerical and categorical features. Categorical features must be
			encoded with an integer labeling (ex: ['cat', 'dog', 'cat', 'bird']
			is encoded as [0, 1, 0, 2]).

		Returns
		=======
		x_transformed : tensor
			A tensor for a transformed version of the input data where categorical
			features are replaced with an embedded representation. The shape of
			this tensor is:

			(num_samples, num_numerical_features + sum_embedding_dimensions)

			The order of features will also be re-arranged, with all embedded
			features appearing after (or to the right of) numerical features.

		"""
		
		# Get column indices for categorical and numerical features
		num_features = x.get_shape()[1]
		all_indices = set(range(num_features))
		embedding_indices = set(self.embedding_map.keys())
		identity_indices = all_indices - embedding_indices
		
		# Convert sets to lists
		identity_indices = list(identity_indices)
		embedding_indices = list(embedding_indices)
		
		# May need to consider using tf.dynamic_partition instead of
		# tf.gather if memory becomes an issue
		# Data that will not be transformed
		x_identity = tf.gather(x, indices=identity_indices, axis=1)
		
		# Collected embedded representations
		embedded_features = list()

		for index, embedding_matrix in self.embedding_map.items():
			feature = tf.gather(x, indices=index, axis=1)
			feature = tf.cast(feature, dtype=tf.int32)
			embedded_feature = tf.nn.embedding_lookup(embedding_matrix, feature)
			embedded_features.append(embedded_feature)
		
		# Merge non-embedded and embedded data
		x_transformed = tf.concat(
			[x_identity] + embedded_features, axis=1, name='x_transformed')
		return x_transformed

	def calc_num_outputs(self, num_inputs):
		num_categorical_features = len(self.embedding_map)
		num_outputs = num_inputs \
			- num_categorical_features \
			+ self.total_embedding_dimensions
		return num_outputs