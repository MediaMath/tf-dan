import tensorflow as tf

from dan.base import Network

class Discriminator(Network):

	def __init__(
		self, hidden_nodes, activation=tf.nn.relu,
		batch_norm=True, dropout=0.0):
		"""
		Defines the graph for the discriminator network.
		
		Parameters
		==========
		x : tensor, shape = (num_samples, num_features)
			Negative and positive samples or disguised samples.
		
		Returns
		=======
		logits : tensor, shape = (num_samples, 2)
			Output of the final discriminator layer with no activation applied.
		"""

		super().__init__(
			'discriminator', hidden_nodes, activation, batch_norm, dropout)

		output_layer = tf.layers.Dense(2, name='logits')
		self.layers.append(output_layer)

	def __call__(self, x):
		"""
		Parameters
		==========
		x : tensor, shape = (num_samples, num_features)
			Either true samples or disguised samples (but not both).
		
		Returns
		=======
		logits : tensor, shape = (num_samples, 2)
			Output of the final discriminator layer with no activation applied.
		"""
		return super().__call__(x)