import tensorflow as tf

class Network:

	def __init__(
		self, name, hidden_nodes, activation=tf.nn.relu,
		batch_norm=True, dropout=0.0):
		"""
		Base callable class for both Disguise and Discriminator networks.

		Parameters
		==========
		name : str, one of {'disguise', 'discriminator'}
			The name used to define the variable scope of all weights within
			the network.
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

		self.name = name
		self.activation = activation
		self.num_hidden_layers = len(hidden_nodes)

		self.layers = list()

		for num_nodes in hidden_nodes:
			dense_layer = tf.layers.Dense(
				num_nodes, activation=self.activation)
			self.layers.append(dense_layer)
			if batch_norm:
				self.layers.append(tf.layers.BatchNormalization())
			if dropout > 0.0:
				self.layers.append(tf.layers.Dropout(dropout))

	def __call__(self, x):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			for layer in self.layers:
				x = layer(x)
		return x