# -*- coding: utf-8 -*-
# model training script for Kuala Lumpur real estate price prediction
# adapted from notebook version https://github.com/ryanmark1867/deep_learning_best_practices/blob/master/notebooks/model_training_keras_preprocessing.ipynb 
# using env. variable handling adapted from https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/edition2/10_mlops/model.py

import time
start_time = time.time()

"""## Import libraries and configs"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# carry over imports from custom layer version
import time
# import datetime, timedelta
import datetime
##import pydotplus
from datetime import datetime, timedelta
from datetime import date
from dateutil import relativedelta
from io import StringIO
import pandas as pd
import pickle
from pickle import dump
from pickle import load
#from sklearn.base import BaseEstimator
#from sklearn.base import TransformerMixin
# DSX code to import uploaded documents
from io import StringIO
import requests
import json
#from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
### remove this include in container version
#from sklearn.model_selection import train_test_split
##import matplotlib.pyplot as plt
# %matplotlib inline
import os
import yaml
import math
import argparse
import os
import logging
import argparse
from google.cloud import storage
import fnmatch

logging.getLogger().setLevel(logging.INFO)
logging.info("tensorflow version is: "+str(tf.__version__))

# load config file
current_path = os.getcwd()
logging.info("current directory is: "+current_path)
# function to translate from yaml to argparser
def create_argparser_from_yaml(yaml_data):
    parser = argparse.ArgumentParser()
    
    for key, value in yaml_data.items():
        arg_type = type(value)
        if arg_type == bool:
            action = 'store_false' if value else 'store_true'
            parser.add_argument(f'--{key}', dest=key, action=action, default=value)
        else:
            parser.add_argument(f'--{key}', dest=key, type=arg_type, default=value)
    
    return parser
logging.info("ABOUT TO argparse")
# get the arguments passed from the pipeline script
parser = argparse.ArgumentParser()
# there is a single argument - the URI of the config yaml file that contains the training parameters
parser.add_argument(
        '--config_bucket',
        help='Config details',
        required=True
    )
args = parser.parse_args().__dict__
logging.info("DICT argparse")
config_bucket = args['config_bucket']
logging.info("config_bucket is: "+str(config_bucket))
# use the method described here to get parts of URI https://engineeringfordatascience.com/posts/how_to_extract_bucket_and_filename_info_from_gcs_uri/
bucket_name = config_bucket.split("/")[2]
object_name = "/".join(config_bucket.split("/")[3:])
# read the object https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage
storage_client2 = storage.Client()
# get objects for the bucket name and file path within the bucket
bucket = storage_client2.bucket(bucket_name)
blob_out = bucket.blob(object_name)
# define the name for the file in the container file system
destination_file_name = 'config.yml'
logging.info("bucket_name is: "+str(bucket_name))
logging.info("object_name is: "+str(object_name))
logging.info("destination_file_name is: "+str(destination_file_name))
# download the config file to a file in the container
blob_out.download_to_filename(destination_file_name)
# load the config file into the dictionary config
try:
    with open (destination_file_name, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file')


logging.info("config is: "+str(config))

# load parameters

repeatable_run = config['test_parms']['repeatable_run']
# fix seeds to get identical results on mulitiple runs
if repeatable_run:
    from numpy.random import seed
    seed(4)
    tf.random.set_seed(7)

logging.info("ABOUT TO ASSIGN CONFIG")
testproportion = config['test_parms']['testproportion'] # proportion of data reserved for test set
trainproportion = config['test_parms']['trainproportion'] # proportion of non-test data dedicated to training (vs. validation)
get_test_train_acc = config['test_parms']['get_test_train_acc']
verboseout = config['general']['verboseout']
includetext = config['general']['includetext'] # switch to determine whether text columns are included in the model
save_model_plot = config['general']['save_model_plot'] # switch to determine whether to generate plot with plot_model
tensorboard_callback = config['general']['tensorboard_callback'] # switch to determine if tensorboard callback defined

presaved = config['general']['presaved']
savemodel = config['general']['savemodel']
picklemodel = config['general']['picklemodel']
hctextmax = config['general']['hctextmax']
maxwords = config['general']['maxwords']
textmax = config['general']['textmax']

targetthresh = config['general']['targetthresh']
targetcontinuous = config['general']['targetcontinuous']
target_col = config['general']['target_col']



emptythresh = config['general']['emptythresh']
zero_weight = config['general']['zero_weight']
one_weight = config['general']['one_weight']
one_weight_offset = config['general']['one_weight_offset']
patience_threshold = config['general']['patience_threshold']


# modifier for saved model elements
modifier = config['general']['modifier']


# default hyperparameter values
learning_rate = config['hyperparameters']['learning_rate']
dropout_rate = config['hyperparameters']['dropout_rate']
l2_lambda = config['hyperparameters']['l2_lambda']
loss_func = config['hyperparameters']['loss_func']
output_activation = config['hyperparameters']['output_activation']
batch_size = config['hyperparameters']['batch_size']
epochs = config['hyperparameters']['epochs']

# date values
date_today = datetime.now()
print("date today",date_today)

# pickled original dataset and post-preprocessing dataset
pickled_data_file = config['general']['pickled_data_file']
pickled_dataframe = config['general']['pickled_dataframe']

# experiment parameter

current_experiment = config['test_parms']['current_experiment']

# load lists of column categories
collist = config['categorical']
textcols = config['text']
continuouscols = config['continuous']
excludefromcolist = config['excluded']
logging.info("THROUGH ASSIGN CONFIG")
"""## Helper functions"""

# get the paths required

def get_path():
    '''get the path for data files

    Returns:
        path: path for data files
    '''
    rawpath = os.getcwd()
    # data is in a directory called "data" that is a sibling to the directory containing the notebook
    path = os.path.abspath(os.path.join(rawpath, 'data'))
    return(path)

def get_pipeline_path():
    '''get the path for pipeline files
    
    Returns:
        path: path for pipeline files
    '''
    rawpath = os.getcwd()
    # data is in a directory called "data" that is a sibling to the directory containing the notebook
    path = os.path.abspath(os.path.join(rawpath, '..', 'pipelines'))
    return(path)

def get_model_path():
    '''get the path for model files
    
    Returns:
        path: path for model files
    '''
    rawpath = os.getcwd()
    # data is in a directory called "data" that is a sibling to the directory containing the notebook
    path = os.path.abspath(os.path.join(rawpath, '..', 'models'))
    return(path)

def set_experiment_parameters(experiment_number, count_no_delay, count_delay):
    ''' set the appropriate parameters for the experiment 
    Args:
        experiment_number: filename containing config parameters
        count_no_delay: count of negative outcomes in the dataset
        count_delay: count of positive outcomes in the dataset

    Returns:
        early_stop: whether the experiment includes an early stop callback
        one_weight: weight applied to positive outcomes
        epochs: number of epochs in the experiment
        es_monitor: performance measurement tracked in callbacks
        es_mod: direction of performance being tracked in callbacks
    
    '''
    print("setting parameters for experiment ", experiment_number)
    # default settings for early stopping:
    es_monitor = "val_loss"
    es_mode = "min"
    if experiment_number == 0:
        #
        early_stop = False
        #
        one_weight = 1.0
        #
        epochs = 1
    elif experiment_number == 9:
        #
        early_stop = True
        es_monitor="val_accuracy"
        es_mode = "max"
        #
        one_weight = (count_no_delay/count_delay) + one_weight_offset
        #
        get_test_train_acc = False
        #
        epochs = 20    
    elif experiment_number == 1:
        #
        early_stop = False
        #
        one_weight = 1.0
        #
        epochs = 10
    elif experiment_number == 2:
        #
        early_stop = False
        #
        one_weight = 1.0
        #
        epochs = 50
    elif experiment_number == 3:
        #
        early_stop = False
        #
        one_weight = (count_no_delay/count_delay) + one_weight_offset
        #
        epochs = 50
    elif experiment_number == 4:
        #
        early_stop = True
        es_monitor = "val_loss"
        es_mode = "min"
        #
        one_weight = (count_no_delay/count_delay) + one_weight_offset
        #
        epochs = 50
    elif experiment_number == 5:
        #
        early_stop = True
        # if early stopping fails because the level of TensorFlow/Python, comment out the following
        # line and uncomment the subsequent if statement
        es_monitor="val_accuracy"
        '''
        if sys.version_info >= (3,7):
            es_monitor="val_accuracy"
        else:
            es_monitor = "val_acc"
        '''
        es_mode = "max"
        #
        one_weight = (count_no_delay/count_delay) + one_weight_offset
        #
        epochs = 100
    else:
        early_stop = True
    return(early_stop, one_weight, epochs,es_monitor,es_mode)

"""## Load the cleaned up dataset


"""

def assign_container_env_variables():
    """
    Copy the environment variables set in the container by the Vertex AI SDK
    """
    # Copy values from environment variables
    OUTPUT_MODEL_DIR = os.getenv("AIP_MODEL_DIR")  # or None
    TRAIN_DATA_PATTERN = os.getenv("AIP_TRAINING_DATA_URI")
    EVAL_DATA_PATTERN = os.getenv("AIP_VALIDATION_DATA_URI")
    TEST_DATA_PATTERN = os.getenv("AIP_TEST_DATA_URI")
    logging.info("patterns train: "+str(TRAIN_DATA_PATTERN))
    logging.info("patterns val: "+str(EVAL_DATA_PATTERN))
    logging.info("patterns test: "+str(TEST_DATA_PATTERN))
    return OUTPUT_MODEL_DIR, TRAIN_DATA_PATTERN, EVAL_DATA_PATTERN, TEST_DATA_PATTERN

def ingest_data(tracer_pattern,target_col):
    '''load list of valid routes and directions into dataframe
    Args:
        tracer_pattern: one of TRAIN_DATA_PATTERN, EVAL_DATA_PATTERN, TEST_DATA_PATTERN
        target_col: the target column
    
    Returns:
        merged_data: dataframe loaded from patterns
    '''
    logging.info("in ingest_data for"+str(tracer_pattern))
    logging.info("target_col is: "+str(target_col))

    # use the pattern to find all the bucket files and create a df from it
    # parse the URI into the bucket name and blob pattern
    bucket_pattern = tracer_pattern.split("/")[2]
    pattern = "/".join(tracer_pattern.split("/")[3:])
    pattern_client = storage.Client()
    bucket = pattern_client.get_bucket(bucket_pattern)
    # get all the blobs in the bucket
    blobs = bucket.list_blobs()
    # get the URIs for all the blobs (files) that match the pattern
    matching_files = [f"gs://{bucket_pattern}/{blob.name}" for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]
    logging.info("matching_files is: "+str(matching_files))
    # build a dataframe from all the matching files
    merged_data = pd.concat([pd.read_csv(f) for f in matching_files], ignore_index=True)
    logging.info(" merged_data shape: "+str(merged_data.shape))
    # remove spaces from and lowercase column names (to avoid model saving issues)
    merged_data.columns = merged_data.columns.str.replace(' ', '_')
    merged_data.columns  = merged_data.columns.str.lower()
    # set the target column in the dataframe
    merged_data['target'] = np.where(merged_data[target_col] >= merged_data[target_col].median(), 1, 0 )
    return(merged_data)



"""## Control cell for ingesting cleaned up dataset"""

# control cell for ingesting data
logging.info("ABOUT TO GET ENV VARIABLES")
# load the environment variables set by the Vertex AI SDK
OUTPUT_MODEL_DIR, TRAIN_DATA_PATTERN, EVAL_DATA_PATTERN, TEST_DATA_PATTERN = assign_container_env_variables()
# get dataframes for each subset of the dataset
target_col = target_col.lower()
logging.info("target_col is: "+target_col)
train = ingest_data(TRAIN_DATA_PATTERN,target_col)
val = ingest_data(EVAL_DATA_PATTERN,target_col)
test = ingest_data(TEST_DATA_PATTERN,target_col)
logging.info("ASSIGNED train val test")

t_shape = train.shape
logging.info("train shape "+str(t_shape))
logging.info("PAST train shape")
v_shape = val.shape
logging.info("val shape "+str(v_shape))
logging.info("PAST val shape")
te_shape = test.shape
logging.info("test shape "+str(te_shape))
logging.info("PAST test shape")
# update column name lists to match updates made to column names in dataset
config['categorical'] = [x.replace(" ", "_") for x in config['categorical']]
config['continuous'] = [x.replace(" ", "_") for x in config['continuous']]
config['categorical'] = [x.lower() for x in config['categorical']]
config['continuous'] = [x.lower() for x in config['continuous']]
logging.info("PAST col fixup")
count_no_delay = train[train['target']==0].shape[0]
count_delay = train[train['target']==1].shape[0]
logging.info("PAST count delay")
# define parameters for the current experiment
experiment_number = current_experiment
early_stop, one_weight, epochs,es_monitor,es_mode = set_experiment_parameters(experiment_number, count_no_delay, count_delay)
logging.info("one_weight is "+str(one_weight))
logging.info("epochs is "+str(epochs))
logging.info("es_monitor is "+str(es_monitor))
logging.info("es_mode is "+str(es_mode))




"""## Split the DataFrame into training, validation, and test sets

"""





logging.info(str(len(train))+'training examples')
logging.info(str(len(val))+'validation examples')
logging.info(str(len(test))+'test examples')

"""## Create an input pipeline using tf.data

Utility function that converts each training, validation, and test into a `tf.data.Dataset`, then shuffles and batches the data.
"""

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('target')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

"""Use `df_to_dataset` to check the format of the data the input pipeline helper function returns by calling it on the training data, and use a small batch size to keep the output readable:"""


train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of Sizes:', train_features['size'][0:10])
print('A batch of targets:', label_batch[0:10] )

"""## Apply the Keras preprocessing layers

The Keras preprocessing layers allow you to build Keras-native input processing pipelines, which can be used in the training process and also at inference time. In this notebook, we will be replacing the pipelines based on scikit-learn that you saw in earlier chapters of this book with a Keras-native approach. In other words, we will be taking advantage of the tabular data capabilities in Keras instead of creating them from scratch ourselves.

In this notebook, you will use the following four preprocessing layers to demonstrate how to perform preprocessing, structured data encoding, and feature engineering (from https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers):

- `tf.keras.layers.Normalization`: Performs feature-wise normalization of input features.
- `tf.keras.layers.CategoryEncoding`: Turns integer categorical features into one-hot, multi-hot, or <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" class="external">tf-idf</a>
dense representations.
- `tf.keras.layers.StringLookup`: Turns string categorical values into integer indices.
- `tf.keras.layers.IntegerLookup`: Turns integer categorical values into integer indices.

### Continuous columns

For each continuous column, use a `tf.keras.layers.Normalization` layer to standardize the distribution of the data.

Define a new utility function that returns a layer which applies feature-wise normalization to continuous columns using that Keras preprocessing layer:
"""

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

"""Next, test the new function by calling it to normalize the `size` column:"""

size_col = train_features['size']
layer = get_normalization_layer('size', train_ds)
# examine the first few elements
layer(size_col)[0:10]

"""### Categorical columns

For categorical columns, define another new utility function that returns a layer which maps values from a vocabulary to integer indices and multi-hot encodes the features using the `tf.keras.layers.StringLookup`, `tf.keras.layers.IntegerLookup`, and `tf.keras.CategoryEncoding` preprocessing layers:
"""

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))

"""Test the `get_category_encoding_layer` function by calling it on the `furnishing` column to turn it into multi-hot encoded tensors:"""

test_type_col = train_features['furnishing']
test_type_layer = get_category_encoding_layer(name='furnishing',
                                              dataset=train_ds,
                                              dtype='string')
test_type_layer(test_type_col)

"""Repeat the process on the `property_type` column.

Note that we needed to update the column names to lowercase them and to replace spaces with underscores to avoid errors when we save the trained model.
"""

test_age_col = train_features['property_type']
test_age_layer = get_category_encoding_layer(name='property_type',
                                             dataset=train_ds,
                                             dtype='string',
                                             max_tokens=5)
test_age_layer(test_age_col)

"""## Preprocess the colums that will be used to train the model

Next, apply the preprocessing utility functions to the continuous and categorical features.

The columns that are used to train the model are specified in the config file model_training_config.yml, ingested as `config['continuous']`, the list of continuous columns, and `config['categorical']`, the list of categorical columns.
"""

# transform the dataframe for each dataset subset into a tf dataset
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

"""Normalize the continuous columns, and add them to one list of inputs called `encoded_features`:"""

all_inputs = []
encoded_features = []

# Continuous features.
for header in config['continuous']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

"""Repeat the same step for categorical columns:"""

for header in config['categorical']:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)

encoded_features

all_inputs

"""## Define and train the model

Use the [Keras Functional API](https://www.tensorflow.org/guide/keras/functional) to define the model layers. For the first layer in your model, merge the list of feature inputs—`encoded_features`—into one vector via concatenation with `tf.keras.layers.concatenate`.
"""

all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(dropout_rate)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

"""Compile the model using hyperparameters defined in the config file."""

model.compile(optimizer=config['hyperparameters']['optimizer'],
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=config['metrics'])

"""Create and save a visualization of the model.

"""

# Use `rankdir='LR'` to make the graph horizontal.
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
# remove for container version
##tf.keras.utils.plot_model(model, show_shapes=True)

# set up early stopping
def set_early_stop(es_monitor, es_mode):
    ''' given monitoring parameter es_monitor and mode es_mode, define early stopping callback, save model callback, and 
    TensorBoard callback
    
    Args:
        es_monitor: the performance parameter to monitor in the callback
        es_mode: the extremity (max or min) to optimize towards
        
    Returns:
        callback_list: list of callback objects
        save_model_path: fully qualified filename to save optimal model to  
    
    
    '''
    # define callback for early stopping
    callback_list = []
    es = EarlyStopping(monitor=es_monitor, mode=es_mode, verbose=1,patience = patience_threshold)
    callback_list.append(es)
    #model_path = get_model_path()
    #save_model_path = os.path.join(model_path,'scmodel'+modifier+"_"+str(experiment_number))
    # define callback to save best model
    #mc = ModelCheckpoint(save_model_path, monitor=es_monitor, mode=es_mode, verbose=1, save_best_only=True, save_format='tf')
    #callback_list.append(mc)
    # define callback for TensorBoard
    if tensorboard_callback:
        tensorboard_log_dir = os.path.join(get_path(),"tensorboard_log",datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(log_dir= tensorboard_log_dir)
        callback_list.append(tensorboard)
    return(callback_list)

"""Train the model"""

@tf.autograph.experimental.do_not_convert
def fit_model():
    if config['general']['early_stop']:
       callback_list = set_early_stop(es_monitor, es_mode)
       model.fit(train_ds, epochs=config['hyperparameters']['epochs'], validation_data=val_ds,callbacks=callback_list)
    else:
       model.fit(train_ds, epochs=config['hyperparameters']['epochs'], validation_data=val_ds)

fit_model()

loss, accuracy = model.evaluate(test_ds)
logging.info("Test Loss"+str(loss))
logging.info("Test Accuracy"+str(accuracy))

"""## Use the trained model to get predictions on new data points

Now that the model has been trained, thanks to the Keras preprocessing layers, we can use it to get predictions for new data points.

We will [save and reload the Keras model](../keras/save_and_load.ipynb) with `Model.save` and `Model.load_model` before doing inference on new data:
"""

# save the trained model to the location passed by the Vertex AI SDK
tf.saved_model.save(model, OUTPUT_MODEL_DIR)

logging.info("categorical columns: "+str(config['categorical']))
logging.info("continuous columns: "+str(config['continuous']))



#logging.info("--- %s seconds ---" % (time.time() - start_time))