# test harness to try out aspects of the Vertex AI ML pipeline creation

# imports
import os
import logging
import argparse
from google.cloud import storage
import yaml
import glob
import pandas as pd
import fnmatch

# set variables
logging.getLogger().setLevel(logging.INFO)
project_id = 'first-project-ml-tabular'
pipeline_root_path = '/home/ryanmark2023/ml_pipeline'

# build argparser and use it to ingest the command line argument
parser = argparse.ArgumentParser()
parser.add_argument(
        '--config_bucket',
        help='Config details',
        required=True
    )
args = parser.parse_args().__dict__
config_bucket = args['config_bucket']

# arg to use on command line: "gs://third-project-ml-tabular-bucket/training_scripts/model_training_config.yml"

# use the method described here to get parts of Google Cloud Storage URI 
# https://engineeringfordatascience.com/posts/how_to_extract_bucket_and_filename_info_from_gcs_uri/
bucket_name = config_bucket.split("/")[2]
object_name = "/".join(config_bucket.split("/")[3:])

# read the object using the approach documented here
# https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob_out = bucket.blob(object_name)
print("config_bucket",config_bucket)
print("bucket_name",bucket_name)
print("object_name",object_name)

# read the config file from the storage bucket
with blob_out.open("r") as f:
    config = yaml.safe_load(f)
print("config is: ",config)
stringer = blob_out.download_as_string()
stringer_string = stringer.decode("utf-8")
print("stringer_string is: ", stringer_string)

# gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/test-00000-of-00004.csv
#tracer_pattern = "gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/training-00001-of-00004.csv"

# set an example pattern (same format at the AIP variables like TRAIN_DATA_PATTERN that are used to
# pass training details to the training container
tracer_pattern = "gs://second-project-ml-tabular-bucket/staging/aiplatform-custom-training-2023-04-04-14:53:53.021/dataset-2901415472631119872-tables-2023-04-04T14:53:53.487135Z/test-*.csv"
# break down the pattern between the bucket and blob
bucket_pattern = tracer_pattern.split("/")[2]
pattern = "/".join(tracer_pattern.split("/")[3:])
print("pattern is: ",pattern)

pattern_client = storage.Client()
bucket = pattern_client.get_bucket(bucket_pattern)
blobs = bucket.list_blobs()
# get the list of URIs that match the pattern
matching_files = [f"gs://{bucket_pattern}/{blob.name}" for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]
print("matching_files is: ",matching_files)
# create a dataframe that combines all the files that match the pattern
df = pd.concat([pd.read_csv(f) for f in matching_files], ignore_index=True)
print("df shape is",df.shape)

