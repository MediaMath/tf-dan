# Disguise Adversarial Networks for Imbalanced Ad Conversion Datasets

For a high-level overview of this project, please see our [blog post](https://data-science-mediamath.ghost.io/p/0835146f-894b-4aca-9ebf-5046c9b5da08/).

This repository contains the codebase for our implementation of the [Disguise Adversarial Network (DAN)](https://www.ijcai.org/proceedings/2017/0220.pdf), which was trained and evaluated on MediaMath's advertising campaign data. Though these datasets will not be made publically available, we nonetheless provide code for how we download, process, and ingest this data for our DAN experiments.

**A note about incomplete and deprecated scripts**: A number of scripts are either incomplete or do not work. As a record of directions we attempted, these scripts were moved from **/scripts/** to **/scripts/incomplete/**. An additional number of scripts are either older versions or redundant variations of other scripts. As such, they were moved from **/scripts/** to **/scripts/deprecated/**.

## Project Organization

------------
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── dan                <- Disguise Adversial Network package.
    │   ├── __init__.py   
    │   ├── base.py
    │   ├── dan.py
    │   ├── data.py
    │   ├── discriminator.py
    │   ├── disguise.py
    │   ├── embedding.py
    │   └── synthetic_data.py
    │
    ├── data
    │   ├── intermediate   <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a two digit number (for ordering),
    │                         the creator's initials, and a short `-` delimited description,
    │                         e.g. `01-dz-initial-data-exploration`.
    │
    ├── reports            <- Analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting.
    │   ├── final-report
    │   ├── paper-summary
    │   ├── poster
    │   ├── progress-report-1
    │   ├── progress-report-2
    │   └── tensorboard-csv
    │
    ├── scripts            <- Data processing and experimentation scripts
    │   ├── deprecated     <- Minor variations or older versions of current scripts
    │   └── incomplete     <- Incomplete and non-functional scripts
    │
    └── tests              <- Pytest testing suites
--------

## Usage

### Downloading the data

```
cd scripts/download_mediamath.sh
sh download_mediamath.sh
```

The above will place the CTR and CPC datasets in **data/raw/mm-ctr/** and **data/raw/mm-cpc/**, respectively.

### Processing CPC data

Preprocessing scripts enable two ways to represent the CPC data (dense categorical embeddings and one-hot encodings) and only one for the CTR data (dense categorical embeddings). We document only CPC data preprocessing as the procedure for CTR embedding-based preprocessing is largely the same as that for CPC.

For data whose categorical values are represented via dense embedding vectors, the entire dataset is loaded into memory before being split into batches for model training. For data whose categorical values are represented via one-hot encodings, a custom data generator is used to read raw data from disk and serve one-hot encoded data in small batches to the DAN.

#### For embeddings

To process the CPC dataset for embeddings, run 
```
python process-cpc-data-embedding.py
```

It will apply various cleaning, encoding, and scaling transformations to the raw CSV GZIP data (train, validation, and test) and convert them to model ingestible form. For information on flags and defaults, run ```python process-cpc-data-embedding.py --help```.

The processed validation and test data are stored at **data/processed/mm-cpc/[validation & test].csv** The training data is stored at **intermediate/mm-cpc/train-[negative & positive].csv** since it needs to be sharded.

To shard any set of CSVs, run ```python shard-cpc-data.py``` with appropriate flags. Example:

```
python shard-cpc-data.py --source_dir ../data/intermediate/mm-cpc/ --destination_dir ../data/processed/mm-cpc/ --source_files train-positive.csv train-negative.csv --num_shards 100
```
The above takes CSVs named **train-positive.csv** and **train-negative.csv** located in **../data/intermediate/mm-cpc/** and saves sharded versions at **../data/processed/mm-cpc/**.

Because the validation set consists of 16% positive samples while the train and test sets only consist of 11%, run the following to rebalance the class distribution in the validation set (replaces old validation CSV with new one).

```
python adjust-validation-imbalance-embedding.py --dataset mm-cpc
```

#### For one-hot

To process the CPC dataset with one-hot representations, run 
```
python process-cpc-data-onehot.py --raw_dir /path/to/directory/ --output_dir /path/to/directory
```
where ```--raw_dir``` points to a directory containing separate folders raw train, validation, and test data, and ```--output_dir``` points to a directory where folders for processed train, validation, and test data are stored. The training data will be sharded, and two JSONs, **categorical-vocab.json** and **numerical-stats.json** are created and stored with the training shards. These two JSONs will be used for online batch pre-processing of the data when ingestion is routed through custom data generators.

Because the validation set consists of 16% positive samples while the train and test sets only consist of 11%, run the following to rebalance the class distribution in the validation set (replaces old validation CSV with new one).

```
python adjust-validation-imbalance-onehot.py --dataset mm-cpc
```

### Running experiments

#### Training models

To train a new model or further train a saved one, run
```
<training-script>.py --experiment_dir /path/to/directory/
```
where **\<training-script\>.py** refers to one of **run-experiment-cpc-embedding.py**, **run-experiment-cpc-onehot.py**, or **run-experiment-cpc-embedding-imbalanced.py**. The **experiment_dir** is a directory unique to each model that contains a variety of resources and logs needed to reload a model or evaluate its progress. An experiment directory typically contains:
- **checkpoints/**: where Tensorflow session states and metagraphs are saved
- **logs/**: where Tensorboard logs are saved
- **hyperparameters.txt**: a log of hyperparameters that were used for the training run
- **test_results.csv**: a CSV of metrics evaluated from test set

The **experiment_dir** is the only required flag, though additional flags are available to configure model hyperparameters:

```
<training-script>.py --help
```

**run-experiment-cpc-embedding-imbalanced.py** is a modified version of **run-experiment-cpc-embedding.py** that adjusts the training data to include only 1.1% positive class while validation and test set class distributions remain unchanged.

#### Testing models

For data that has been embedded, to generate **test_results.csv**, simply rerun the training script with the same flags as when training, but add ```--test```.

For data that has been one-hot encoded, the **test_results.csv** is automatically generated following a training run. This is because when using the one-hot pipeline, the processed data output by the custom data generator has a slightly different column ordering depending on whether a new DAN is instantiated (as when training) or a saved DAN is loaded from its checkpoint (as when evaluating). Attempts to pinpoint the source of this error were unsuccessful, so as a workaround, we immediately evaluate a model after early stopping concludes its training run, using the model’s most recent state rather than that checkpointed with the highest validation AUROC.

## The DAN module and custom data generators

Please refer to this [readme](https://github.com/edrinea/f18-mediamath/blob/dev/dan/readme.md).
