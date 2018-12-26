import tensorflow as tf

from dan.base import Network

class Disguise(Network):

	def __init__(
		self, num_inputs, hidden_nodes, activation=tf.nn.relu,
		batch_norm=True, dropout=0.0):
		"""
		A callable that defines the architecture of a disguise network.

		Parameters
		==========
		num_inputs : int
			The number of features in each data sample.
		hidden_nodes : array-like of ints
			A list of integers corresponding to the number of nodes in each
			hidden layer.
		activation : function
			The element-wise non-linearity applied before the output of each layer.
		batch_norm : bool
			If True, adds a batch normalization layer after each hidden dense layer.
		dropout : float, between 0 and 1 inclusive
			If greater than 0, adds a dropout regularization layer following each
			hidden dense or batch norm layer.

		"""
		super().__init__(
			'disguise', hidden_nodes, activation, batch_norm, dropout)
		self.num_inputs = num_inputs

		output_layer = tf.layers.Dense(num_inputs, name='disguised')
		self.layers.append(output_layer)
		
		def __call__(self, x):
			"""
			Parameters
			==========
			x : tensor, shape = (num_negatives_in_batch, num_features)
				All negative samples in the current mini-batch.
			
			Returns
			=======
			disguised : tensor, shape = (num_negatives_in_batch, num_features)
				All negative samples in the current mini-batch disguised as 
				positive samples.
			"""
			return super().__call__(x)