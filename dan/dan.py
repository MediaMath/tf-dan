import os
import numpy as np
import tensorflow as tf
import glob

from functools import reduce
from tensorboard.backend.event_processing.event_accumulator \
	import EventAccumulator

class DAN:
		
	def __init__(
		self, disguise, discriminator, checkpoint_dir, log_dir,
		embedding_transformer=None, lambd=1, eta=1, learning_rate=1e-3,
		num_inputs=None, early_stopping=2):
		"""
		Disguise Adversarial Network
		
		Parameters
		==========
		checkpoint_dir : str
			The path to the directory where checkpoints are saved and loaded.
		num_features : int
			The dimensionality of the dataset, i.e. the number of input features.
		lambd : float
			The amount of regularization applied in the disguise network loss.
			Larger values restrict the diguise network from applying drastic
			transformations on negative samples.
		eta : float
			Controls the contribution of the predictive entropy of disguised
			samples to the entire discriminator loss.
		learning_rate : float
			The learning rate of the optimizer.
		"""

		self.embedding_transformer = embedding_transformer
		self.disguise_network = disguise
		self.discriminator_network = discriminator

		if disguise is None and num_inputs is None:
			raise ValueError(
				'If no disguise network, num_inputs must be specified.')
		elif (disguise is not None 
			and embedding_transformer is not None 
			and num_inputs is None):
			raise ValueError(
				'If using embeddings, num_inputs must be specified.')
		elif disguise is not None and embedding_transformer is None:
			self.num_inputs = disguise.num_inputs
		else:
			self.num_inputs = num_inputs

		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir
		self.lambd = lambd
		self.eta = eta
		self.learning_rate = learning_rate
		
		self.sess = tf.Session()
		with tf.variable_scope('dan', reuse=tf.AUTO_REUSE):
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self._create_network()
			self._create_losses()
			self._create_optimizers()
			self._create_metrics()
			self._create_summaries()
		
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.tables_initializer())

		self._load_checkpoint()
		self._best_val_auroc = self._load_best_score()
		self.early_stopping = early_stopping
		self.no_improvement = 0

	def _load_best_score(self):
		logs = glob.glob(os.path.join(self.log_dir, '*'))
		print('Found {} log files.'.format(len(logs)))
		if len(logs) == 1:
			print('Setting best validation AUROC to -inf...')
			return -float('inf')
		else:
			scores = [-float('inf')]
			for log in logs:
				event_accumulator = EventAccumulator(log)
				event_accumulator.Reload()
				event_accumulator.Tags()
				try:
					data = event_accumulator.Scalars('dan/validation_auroc')
				except:
					continue
				data = [event.value for event in data]
				scores.extend(data)
			best_score = max(scores)
			print('Setting best validation AUROC to {:.4f}'.format(best_score))
			return best_score
		
	def _load_checkpoint(self):
		"""
		Checks if a model checkpoint exists, and if so, alters 'self.sess' to
		reflect the appropriate state of the Tensorflow computation graph.
		"""
		self.saver = tf.train.Saver(max_to_keep=3)
		checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
	
		# If checkpoint exists and is reachable, load checkpoint state into 'sess'
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print('loaded checkpoint: {}'.format(checkpoint.model_checkpoint_path))
		else:
			print(
				'Could not find old checkpoint. '
				'Creating new checkpoint directory.'
			)
			if not os.path.exists(self.checkpoint_dir):
				os.mkdir(self.checkpoint_dir)

	def _save_checkpoint(self):
		self.saver.save(
			self.sess,
			self.checkpoint_dir,
			global_step=self.global_step)
		
	def _create_network(self):
		"""
		Defines the graph for the entire DAN.
		"""
		
		# Tensors whose values will be provided during execution
		self.x_input = tf.placeholder(
			dtype=tf.float32,
			shape=(None, self.num_inputs),
			name='x_input')
		self.y_input = tf.placeholder(
			dtype=tf.float32,
			shape=(None, 2),
			name='y_input')

		if self.embedding_transformer is None:
			self.x_input_transformed = tf.identity(self.x_input)
		else:
			self.x_input_transformed = self.embedding_transformer(self.x_input)

		# Indices of negative samples in current mini-batch
		self.negative_indices = tf.placeholder(
			dtype=tf.int32, shape=(None,), name='negative_indices')
		
		if self.disguise_network is not None:
			
			# Filter for negative samples and disguise
			self.x_negatives = tf.gather(
				self.x_input_transformed, self.negative_indices)
			self.disguised = self.disguise_network(self.x_negatives)

			# Discriminator outputs for disguised samples
			self.unlabeled_logits = self.discriminator_network(self.disguised)
			self.unlabeled_probas = tf.nn.softmax(self.unlabeled_logits)
			self.unlabeled_preds = tf.cast(self.unlabeled_probas >= 0.5, tf.float32)
		
		# Discriminator outputs for true samples
		self.labeled_logits = self.discriminator_network(self.x_input_transformed)
		self.labeled_probas = tf.nn.softmax(self.labeled_logits)
		self.labeled_preds = tf.cast(self.labeled_probas >= 0.5, tf.float32)
		
	def _create_losses(self):
		"""
		Defines discriminator and disguise loss functions and their respective
		components. tf.reduce_mean is applied to each component individually
		for ease of Tensorboard monitoring.
		"""

		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=self.y_input, logits=self.labeled_logits)
		self.cross_entropy = tf.reduce_mean(cross_entropy)
		
		if self.disguise_network is not None:
			predictive_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
				labels=self.unlabeled_probas, logits=self.unlabeled_logits)
			self.predictive_entropy = tf.reduce_mean(predictive_entropy)
			
			self.discriminator_loss = self.cross_entropy + self.eta * self.predictive_entropy
			
			# The disguise loss is defined only on positive prediction probabilities
			unlabeled_probas_temp = tf.gather(self.unlabeled_probas, 1, axis=1)
			self.disguise_quality = tf.reduce_mean(tf.log(unlabeled_probas_temp))
			self.regularizer = tf.reduce_mean(
				tf.norm(self.disguised - self.x_negatives, ord=1, axis=1))
			self.disguise_loss = -self.disguise_quality + self.lambd * self.regularizer

	def _create_metrics(self):
		with tf.variable_scope('metrics', reuse=tf.AUTO_REUSE):
			self.accuracy, self.accuracy_update = tf.metrics.accuracy(
				labels=self.y_input, predictions=self.labeled_preds, name='accuracy')
			self.auroc, self.auroc_update = tf.metrics.auc(
				labels=self.y_input, predictions=self.labeled_probas,
				curve='ROC', name='auroc')

		running_vars = tf.get_collection(
			tf.GraphKeys.LOCAL_VARIABLES, scope='dan/metrics')
		self.running_vars_initializer = tf.variables_initializer(
			var_list=running_vars)

		if self.disguise_network is not None:
			batch_size = tf.shape(self.unlabeled_preds)[0]
			predicted_positives = tf.gather(self.unlabeled_preds, 1, axis=1)
			num_pred_positives = tf.reduce_sum(predicted_positives)
			self.disguise_success_rate = \
				num_pred_positives / tf.cast(batch_size, tf.float32)

	def _create_summaries(self):

		scalar = tf.summary.scalar
		self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

		# Collects scalar tensors to be later converted into summary operations
		batch_summary_ops = list()
		
		batch_summary_ops.append(
			scalar('cross entropy', self.cross_entropy))

		if self.disguise_network is not None:
			batch_summary_ops.append(
				scalar('predictive entropy', self.predictive_entropy))
			batch_summary_ops.append(
				scalar('discriminator loss', self.discriminator_loss))
			batch_summary_ops.append(
				scalar('disguise quality', self.disguise_quality))
			batch_summary_ops.append(
				scalar('disguise success rate', self.disguise_success_rate))
			batch_summary_ops.append(
				scalar('regularizer', self.regularizer))
			batch_summary_ops.append(
				scalar('disguise loss', self.disguise_loss))

		self.batch_summary_ops = tf.summary.merge(batch_summary_ops)
		
		train_epoch_summary_ops = list()
		val_epoch_summary_ops = list()

		train_epoch_summary_ops.append(
			scalar('train accuracy', self.accuracy))
		train_epoch_summary_ops.append(
			scalar('train auroc', self.auroc))
		self.train_epoch_summary_ops = tf.summary.merge(train_epoch_summary_ops)

		val_epoch_summary_ops.append(
			scalar('validation accuracy', self.accuracy))
		val_epoch_summary_ops.append(
			scalar('validation auroc', self.auroc))
		self.val_epoch_summary_ops = tf.summary.merge(val_epoch_summary_ops)

	def _create_optimizers(self):
		"""
		Defines separate optimizers for the disguise and discriminator networks.
		Each network's optimizer should only update that network's weights despite
		the fact that each network's loss function partly depends on the weights
		of the other network.
		"""
		optimizer = tf.train.AdamOptimizer
		discriminator_weights = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, 'dan/discriminator')
		if self.embedding_transformer is not None:
			embedding_weights = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, 'dan/embeddings')
			# print('embedding_weights:', embedding_weights)
			discriminator_weights += embedding_weights

		# This optimizer is only used when training the DAN without a disguise 
		# network (i.e. discriminator-only mode)
		cross_entropy_optimizer = optimizer(self.learning_rate)
		self.cross_entropy_optimizer = cross_entropy_optimizer.minimize(
			self.cross_entropy,
			var_list=discriminator_weights,
			global_step=self.global_step)

		if self.disguise_network is not None:
			disguise_weights = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, 'dan/disguise')
			disguise_optimizer = optimizer(self.learning_rate)
			self.disguise_optimizer = disguise_optimizer.minimize(
				self.disguise_loss, var_list=disguise_weights)
			
			discriminator_optimizer = optimizer(self.learning_rate)
			self.discriminator_optimizer = discriminator_optimizer.minimize(
				self.discriminator_loss,
				var_list=discriminator_weights,
				global_step=self.global_step)
		
	def _partial_fit(self, batch_x, batch_y, disguise=True):
		"""
		Trains the DAN on a mini-batch of data.
		
		Parameters
		==========
		batch_x : numpy array, shape = (batch_size, num_features)
			A batch of training data.
		batch_y : numpy array, shape = (batch_size, 2)
			A batch of one-hot encoded labels.
		disguise : bool
			If true, the disguise network is trained along with the discriminator.
			Otherwise, the discriminator is optimized only on cross entropy, no
			different from a regular classifier.
		"""
		feed_dict={
			self.x_input: batch_x,
			self.y_input: batch_y,
		}
		negative_indices = np.argwhere(batch_y[:,0]).squeeze()
		if negative_indices.size > 0:
			feed_dict[self.negative_indices] = negative_indices
		else:
			feed_dict[self.negative_indices] = []
		
		if disguise:
			self.sess.run(
				[self.disguise_optimizer, self.discriminator_optimizer],
				feed_dict=feed_dict
			)
			
		else:
			results = self.sess.run(
				self.cross_entropy_optimizer,
				feed_dict=feed_dict
			)

		step = self.sess.run(self.global_step)
		summary_str = self.sess.run(self.batch_summary_ops, feed_dict=feed_dict)
		self.summary_writer.add_summary(summary_str, step)
		
	def fit(
		self, x, y, num_epochs=5, batch_size=256, disguise=True,
		x_val=None, y_val=None):
		"""
		Fits the DAN on the entire training dataset.
		
		x : numpy array, shape = (num_samples, num_features)
			The features of the training set.
		y : numpy array, shape = (num_samples, 2)
			The one-hot encoded labels of the training set.
		num_epochs : int
			The number of times the model is fit on the whole training set.
		batch_size : int
			The number of samples in each mini-batch.
		disguise : bool
			If true, the disguise network is trained along with the discriminator.
			Otherwise, the discriminator is optimized only on cross entropy, no
			different from a regular classifier.
		"""
		if self.disguise_network is None and disguise:
			print(
				'Warning: setting disguise=False since no disguise '
				'network was defined.'
			)
			disguise = False
		
		num_samples = len(y)
		num_batches = num_samples // batch_size

		for epoch in range(num_epochs):
			
			print('Training epoch {}...'.format(epoch))
			shuffled_indices = np.random.permutation(num_samples)	
			x = x[shuffled_indices]	
			y = y[shuffled_indices]	
				
			start_index = 0	
			end_index = batch_size

			for batch in range(num_batches+1):

				batch_x = x[start_index:end_index]
				batch_y = y[start_index:end_index]	
					
				start_index += batch_size	
				end_index += batch_size	
				if end_index > num_samples:	
					end_index = num_samples
					
				self._partial_fit(batch_x, batch_y, disguise)

			step = self.sess.run(self.global_step)
			train_metrics = self.evaluate(x, y, batch_size)
			train_cross_entropy, train_accuracy, train_auroc = train_metrics
			print('Train crossentropy: {:.4}'.format(train_cross_entropy), end='\t')
			print('Train accuracy: {:.4}'.format(train_accuracy), end='\t')
			print('Train AUROC: {:.4}'.format(train_auroc))
			summary_str = self.sess.run(
				self.train_epoch_summary_ops)
			self.summary_writer.add_summary(summary_str, step)

			if x_val is not None and y_val is not None:
				val_metrics = self.evaluate(x_val, y_val, batch_size)
				val_cross_entropy, val_accuracy, val_auroc = val_metrics
				print('Val crossentropy: {:.4}'.format(val_cross_entropy), end='\t')
				print('Val accuracy: {:.4}'.format(val_accuracy), end='\t')
				print('Val AUROC: {:.4}'.format(val_auroc))
				summary_str = self.sess.run(
					self.val_epoch_summary_ops)
				self.summary_writer.add_summary(summary_str, step)

				if val_auroc > self._best_val_auroc:
					print('Improved validation AUROC from {:.4} to {:.4}.'.format(
						self._best_val_auroc, val_auroc))
					print('Saving checkpoint...')
					self._best_val_auroc = val_auroc
					self.no_improvement = 0
					self._save_checkpoint()
				else:
					self.no_improvement += 1
					if self.no_improvement >= self.early_stopping:
						print('Too many epochs with no improvement. Stopping training...')
						break
			else:
				self._save_checkpoint()

	def fit_generator(
		self, train_datagen, num_epochs=10, disguise=True,
		val_datagen=None):

		if self.disguise_network is None and disguise:
			print(
				'Warning: setting disguise=False since no disguise '
				'network was defined.')
			disguise = False

		num_batches = len(train_datagen)
		for epoch in range(num_epochs):
			print('Training epoch {}...'.format(epoch))

			for batch in range(num_batches):
				batch_x, batch_y = self.sess.run([
					train_datagen.x_processed, train_datagen.y_processed])
				self._partial_fit(batch_x, batch_y, disguise)

			step = self.sess.run(self.global_step)

			train_metrics = self.evaluate_generator(train_datagen)
			train_cross_entropy, train_accuracy, train_auroc = train_metrics
			print('Train crossentropy: {:.4}'.format(train_cross_entropy), end='\t')
			print('Train accuracy: {:.4}'.format(train_accuracy), end='\t')
			print('Train AUROC: {:.4}'.format(train_auroc))
			summary_str = self.sess.run(
				self.train_epoch_summary_ops)
			self.summary_writer.add_summary(summary_str, step)

			if val_datagen is not None:
				val_metrics = self.evaluate_generator(val_datagen)
				val_cross_entropy, val_accuracy, val_auroc = val_metrics
				print('Val crossentropy: {:.4}'.format(val_cross_entropy), end='\t')
				print('Val accuracy: {:.4}'.format(val_accuracy), end='\t')
				print('Val AUROC: {:.4}'.format(val_auroc))
				summary_str = self.sess.run(
					self.val_epoch_summary_ops)
				self.summary_writer.add_summary(summary_str, step)

				if val_auroc > self._best_val_auroc:
					print('Improved validation AUROC from {:.4} to {:.4}.'.format(
						self._best_val_auroc, val_auroc))
					print('Saving checkpoint...')
					self._best_val_auroc = val_auroc
					self.no_improvement = 0
					self._save_checkpoint()
				else:
					self.no_improvement += 1
					if self.no_improvement >= self.early_stopping:
						print('Too many epochs with no improvement. Stopping training...')
						break
			else:
				self._save_checkpoint()

	def evaluate(self, x, y, batch_size=64):
		"""
		Evaluates the DAN's classification performance on an entire dataset.
		
		Parameters
		==========
		x : numpy array, shape = (num_samples, num_features)
			The features of the training set.
		y : numpy array, shape = (num_samples, 2)
			The one-hot encoded labels of the training set.
		batch_size : int
			The number of samples in each mini-batch.

		Returns
		=======
		epoch_cross_entropy : float
		epoch_accuracy : float
		epoch_auroc : float
		"""
		num_samples = len(y)
		num_batches = num_samples // batch_size
		
		self.sess.run(self.running_vars_initializer) # Reset metrics

		# shuffled_indices = np.random.permutation(num_samples)
		# x = x[shuffled_indices]
		# y = y[shuffled_indices]

		start_index = 0	
		end_index = batch_size		
		
		cross_entropy_losses = list()
		for batch in range(num_batches+1):
			
			batch_x = x[start_index:end_index]
			batch_y = y[start_index:end_index]

			# Update running variables
			feed_dict = {self.x_input: batch_x, self.y_input: batch_y}
			self.sess.run(
				[self.accuracy_update, self.auroc_update],
				feed_dict=feed_dict)

			# Collect losses
			cross_entropy_loss = self.sess.run(
				self.cross_entropy, feed_dict=feed_dict)
			cross_entropy_losses.append(cross_entropy_loss)

			start_index += batch_size	
			end_index += batch_size	
			if end_index > num_samples:	
				end_index = num_samples
				
		# Calculate epoch metrics
		# Cross entropy is approximate since final batch may have fewer samples
		epoch_cross_entropy = np.mean(cross_entropy_losses)
		epoch_accuracy = self.sess.run(self.accuracy)
		epoch_auroc = self.sess.run(self.auroc)

		return epoch_cross_entropy, epoch_accuracy, epoch_auroc

	def evaluate_generator(self, datagen):
		"""
		Evaluates the DAN's classification performance on an entire dataset.
		
		Parameters
		==========
		datagen : DataGenerator
		batch_size : int
			The number of samples in each mini-batch.

		Returns
		=======
		epoch_cross_entropy : float
		epoch_accuracy : float
		epoch_auroc : float
		"""
		num_batches = len(datagen)
		
		# Reset metrics
		self.sess.run(self.running_vars_initializer)
		
		cross_entropy_losses = list()
		for batch in range(num_batches):
			
			batch_x, batch_y = self.sess.run([
				datagen.x_processed, datagen.y_processed])

			# Update running variables
			feed_dict = {self.x_input: batch_x, self.y_input: batch_y}
			self.sess.run(
				[self.accuracy_update, self.auroc_update],
				feed_dict=feed_dict)

			# Collect losses
			cross_entropy_loss = self.sess.run(
				self.cross_entropy, feed_dict=feed_dict)
			cross_entropy_losses.append(cross_entropy_loss)

		# Calculate epoch metrics
		# Cross entropy is approximate since final batch may have fewer samples
		epoch_cross_entropy = np.mean(cross_entropy_losses)
		epoch_accuracy = self.sess.run(self.accuracy)
		epoch_auroc = self.sess.run(self.auroc)

		return epoch_cross_entropy, epoch_accuracy, epoch_auroc

	def predict_proba(self, x, batch_size=256):
		"""
		Infers the probabilties of the given samples belonging to
		one of two classes.
		
		Parameters
		==========
		x : numpy array, shape = (num_samples, num_features)
		
		Returns
		=======
		probas : numpy array, shape = (num_samples, 2)
		"""
		num_samples = len(x)
		num_batches = num_samples // batch_size
		
		start_index = 0
		end_index = batch_size
		
		probas = list()
		for batch in range(num_batches+1):
			
			batch_x = x[start_index:end_index]

			batch_probas = self._partial_predict_proba(batch_x)
			probas.append(batch_probas)
			
			start_index += batch_size
			end_index += batch_size
			if end_index > num_samples:
				end_index = num_samples

		probas = np.row_stack(probas)
		assert(probas.shape == (num_samples, 2))
		return probas

	def _partial_predict(self, batch_x):
		"""
		Infers the class of the given samples.
		
		Parameters
		==========
		x : numpy array, shape = (num_samples, num_features)
		
		Returns
		=======
		predictions : numpy array, shape = (num_samples, 2)
		"""
		preds = self.sess.run(
			self.labeled_preds, feed_dict={self.x_input: batch_x})
		return preds

	def _partial_predict_proba(self, batch_x):
		probas = self.sess.run(
			self.labeled_probas, feed_dict={self.x_input: batch_x})
		return probas

	def predict(self, x, batch_size=256):
		num_samples = len(x)
		num_batches = num_samples // batch_size
		
		start_index = 0
		end_index = batch_size
		
		predictions = list()
		for batch in range(num_batches+1):
			
			batch_x = x[start_index:end_index]

			batch_pred = self._partial_predict(batch_x)
			predictions.append(batch_pred)
			
			start_index += batch_size
			end_index += batch_size
			if end_index > num_samples:
				end_index = num_samples

		predictions = np.row_stack(predictions)
		assert(predictions.shape == (num_samples, 2))
		return predictions

	def predict_generator(self, datagen):
		num_batches = len(datagen)
		
		predictions = list()
		labels = list()
		for batch in range(num_batches):
			
			batch_x, batch_y = self.sess.run([
				datagen.x_processed, datagen.y_processed])

			batch_pred = self._partial_predict(batch_x)
			predictions.append(batch_pred)
			labels.append(batch_y)

		predictions = np.row_stack(predictions)
		labels = np.row_stack(labels)
		assert(predictions.shape == labels.shape)
		return predictions, labels

	def predict_proba_generator(self, datagen):
		num_batches = len(datagen)
		
		probas = list()
		labels = list()
		for batch in range(num_batches):
			
			batch_x, batch_y = self.sess.run([
				datagen.x_processed, datagen.y_processed])

			batch_probas = self._partial_predict_proba(batch_x)
			probas.append(batch_probas)
			labels.append(batch_y)

		probas = np.row_stack(probas)
		labels = np.row_stack(labels)
		assert(probas.shape == labels.shape)
		return probas, labels

	def transform(self, x_negatives, embed=False):
		"""
		Tranforms the given negative samples into disguised positives.
		
		Parameters
		==========
		x_negatives : numpy array, shape = (num_samples, num_features)
		
		Returns
		=======
		disguised : numpy array, shape = (num_samples, num_features)
		"""
		if self.disguise_network is None:
			raise AttributeError('Cannot call transform without disguise network.')
		negative_indices = list(range(len(x_negatives)))
		disguised = self.sess.run(
			self.disguised, feed_dict={
				self.x_input: x_negatives,
				self.negative_indices: negative_indices}
		)
		return disguised

	def __del__(self):
		self.sess.close()