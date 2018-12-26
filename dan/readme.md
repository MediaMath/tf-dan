# Using the DAN module

```
from dan.dan import DAN
from dan.disguise import Disguise
from dan.discriminator import Discriminator
from dan.embedding import EmbeddingTransformer
from dan.data import DataGenerator
```

The DAN class accepts objects of class Disguise, Discriminator, and EmbeddingTransformer as arguments. We briefly detail the initialization of each component class and then discuss how they can be used in various combinations within a DAN.

### Discriminator

The **Discriminator** class simply defines a densely connected classifier. To define a discriminator network with 2 hidden layers of size 64 and 32:

```
discriminator = Discriminator(hidden_nodes=[64, 32])
```

Note that the number of input features is not required nor the number of output nodes, as the network is always a binary classifier. To add batch normalization, dropout, and a non-ReLU activation after each hidden layer:

```
discriminator = Discriminator(
    hidden_nodes=[64, 32],
    activation=tf.nn.elu,
    batch_norm=True,
    dropout=0.5)
```

### Disguise

The **Disguise** class has the same signature and behavior as the Discriminator class but for one exception: the Disguise class requires an argument for number of input features, as this will also define the number of output features. To define a disguise network that transforms 340-dimensional data with 4 hidden layers of size 256:

```
disguise = Disguise(
    num_inputs=340,
    hidden_nodes=[256, 256, 256, 256],
    activation=tf.nn.elu,
    batch_norm=True,
    dropout=0.5)
```

### EmbeddingTransformer

The **EmbeddingTransformer** builds embedding lookup tables for categorical features that are index encoded. It has two parameters. The first, **index_map**, accepts a dictionary where each key is the column index of a categorical feature and each value is the cardinality of the feature. The second, **alpha**, controls the dimensionality of the embedded feature: embedded_dimensionality = ceil(cardinality ^ (1 / alpha)).

For example, if we know that the 4th and 8th columns of our input data consists of categorical features with cardinalities 10000 and 30000, respectively, and we want them embedded into vectors with sizes that are approximately the cube root of their cardinality:

```
index_map = {4: 10000, 8: 30000}
embedding_transformer = EmbeddingTransformer(index_map=index_map, alpha=3)
```

### DAN initialization
The **DAN** class is initialized differently depending on the situation. In all cases, it requires arguments for where to save the checkpoints (path to a directory) and where to save the Tensorboard logs (path to a directory).

#### As a vanilla binary classifier
When the DAN is initialized with no disguise network, it acts simply as a binary classifier. The **num_inputs** parameter is required so as to properly define a placeholder tensor. To create a classifier for 100-dimensional data:

```
model = DAN(
  num_inputs=100,
  disguise=None,
  discriminator=discriminator,
  checkpoint_dir='path/to/checkpoint/dir/',
  log_dir='path/to/log/dir/')
```

#### As a binary classifier with embedding layers
```
model = DAN(
  num_inputs=100,
  disguise=None,
  discriminator=discriminator,
  embedding_transformer=embedding_transformer,
  checkpoint_dir='path/to/checkpoint/dir/',
  log_dir='path/to/log/dir/')
```

#### As a DAN
Because the disguise network requires the number of input features upon initialization, if the DAN includes a disguise network, **num_inputs** is not required as an argument for the DAN.
```
model = DAN(
  disguise=disguise,
  discriminator=discriminator,
  embedding_transformer=embedding_transformer,
  checkpoint_dir='path/to/checkpoint/dir/',
  log_dir='path/to/log/dir/')
```

#### As a DAN with embedding layers
When using the EmbeddingTransformer with a DAN, the number of inputs to the disguise network is **not** equal to the number of features in the dataset since the disguise network acts on the embedded feature space. Thus, the Disguise class requires the total number of features in embedded space while DAN requires the number of features in pre-embedded space.
```
embedding_transformer = EmbeddingTransformer(index_map=index_map)

disguise = Disguise(
  num_inputs=embedding_transformer.calc_num_outputs(num_inputs),
  hidden_nodes=[256, 256, 256, 256])
  
discriminator = Discriminator(
  hidden_nodes=[64, 32])

dan = DAN(
  disguise, discriminator, CHECKPOINT_DIR, LOG_DIR,
  embedding_transformer=embedding_transformer,
  num_inputs=num_inputs)
```

### DAN usage

#### Memory-based functions

The DAN class exposes only three functions to the user that directly accepts NumPy arrays. These functions are used when it is feasible to load the entire training and validation dataset into memory at once. Let X have shape (batch_size, num_features) and y have shape (batch_size, 2).

To train the DAN regardless of whether it was initialized with a disguise network:

```
dan.fit(X, y, num_epochs=5, batch_size=256)
```

If the DAN was initialized with a disguise network, it can still be trained without it:

```
dan.fit(X, y, num_epochs=5, batch_size=256, disguise=False)
```

To make predictions:

```
dan.predict(X)
```

To disguise samples:
```
dan.transform(X)
```

#### Generator-based functions

To allow the DAN to train and predict on datasets of arbitrarily large sizes, we also expose functions that accept custom data generators as arguments. These generators read in batches of unprocessed data from disk, process the data batch in memory, then supply the processed batch to the DAN.

To instantiate a data generator, three arguments are required:
1. A path to the directory containing CSVs for a given dataset split (train, validation, or test).
2. A dictionary with names of categorical features as keys and lists of all possible unique strings as values. Typically stored on disk as **categorical-vocab.json**.
3. A dictionary with names of continuous features as keys and a nested dictionary with 'mean' and 'std' as keys and their associated floats as values. Typically stored on disk as **numerical-stats.json**.

Note that both dictionaries should only be produced from the training data.

```
train_datagen = DataGenerator(
  csv_paths='path/to/training/csvs/',
  vocab_map=categorical_vocab_dict,
  stats_map=numerical_stats_dict)

val_datagen = DataGenerator(
  csv_paths='path/to/validation/csvs/',
  vocab_map=categorical_vocab_dict,
  stats_map=numerical_stats_dict)
```

To train the dan using data generators and early stopping:

```
dan.fit_generator(train_datagen, val_datagen)
```

To make predictions:
```
dan.predict_generator(test_datagen)
```

To make predictive probability distributions:
```
dan.predict_proba(test_datagen)
```

To return the DAN's cross entropy loss, accuracy, and AUROC
```
dan.evaluate_generator(test_datagen)
```
