# script to (a) drive training process for a Keras model trained on tabular data in Vertex AI, and
# (b) deploy the resulting trained model to a Vertex AI endpoint
# adapts ideas from https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/edition2/10_mlops/train_on_vertexai.py

# imports
from google.cloud import aiplatform
import tensorflow as tf
import argparse
import yaml
import os
from datetime import datetime
import time

# initialize time counter
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")




def get_pipeline_config(path_to_yaml):
    '''ingest the config yaml file
    Args:
        path_to_yaml: yaml file containing parameters for the pipeline script
    
    Returns:
        config: dictionary containing parameters read from the config file
    '''
    print("path_to_yaml "+path_to_yaml)
    try:
        with open (path_to_yaml, 'r') as c_file:
            config = yaml.safe_load(c_file)
    except Exception as e:
        print('Error reading the config file')
    return config
 

# define CustomTrainingJob object
def create_job(config):
    ''' create CustomTrainingJob object
    Args:
        config: dictionary containing parameters read from config file
    
    Returns:
        job: CustomTrainingJob object
    '''
    model_display_name = '{}-{}'.format(config['ENDPOINT_NAME'], TIMESTAMP)
    job = aiplatform.CustomTrainingJob(
            display_name='train-{}'.format(model_display_name),
            script_path = config['script_path'],
            container_uri=config['train_image'],
            staging_bucket = config['staging_path'],
            requirements=['gcsfs'],  # any extra Python packages
            model_serving_container_image_uri=config['deploy_image']
    ) 
    return job


# run job to create dataset
def run_job(job, ds, model_args,config):
    ''' run CustomTrainingJob object, including specifying train/validation/test split of dataset
    Args:
        job: CustomTrainingJob object
        ds: TabularDataSet object associated with managed dataset
        config: dictionary containing parameters read from config file
    
    Returns:
        model: trained Vertex AI model object
    '''
    model_display_name = '{}-{}'.format(config['ENDPOINT_NAME'], TIMESTAMP)
    model = job.run(
        dataset=ds,
        # See https://googleapis.dev/python/aiplatform/latest/aiplatform.html#
        training_fraction_split = config['training_fraction_split'],
        validation_fraction_split = config['validation_fraction_split'],
        test_fraction_split = config['test_fraction_split'],
        model_display_name=model_display_name,
        args=model_args,
        machine_type= config['machine_type']
    )
    return model

def deploy_model(model,config):
    ''' deploy model to a Vertex AI endpoint
    Args:
        model: Vertex AI model object
        config: dictionary containing parameters read from config file
    
    '''
    endpoints = aiplatform.Endpoint.list(
        filter='display_name="{}"'.format(config['ENDPOINT_NAME']),
        order_by='create_time desc',
        project=config['project_id'], 
        location=config['region']
    )
    if len(endpoints) > 0:
        endpoint = endpoints[0]  # most recently created
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=config['ENDPOINT_NAME'], 
            project=config['project_id'], 
            location=config['region']
        )

    # deploy model to a Vertex AI endpoint
    model.deploy(
        endpoint=endpoint,
        traffic_split={"0": 100},
        machine_type=config['machine_type_deploy'],
        min_replica_count=1,
        max_replica_count=1,
    )

if __name__ == '__main__':
    start_time = time.time()
    # load pipeline config parameters
    config = get_pipeline_config('pipeline_config.yml')
    # all the arguments sent to the training script run in the container are sent via
    # a yaml file in Cloud Storage whose URI is the single argument sent
    model_args = ['--config_bucket', config['config_bucket_path']]
    print("model_args: ",model_args)
    # create a CustomTrainingJob object
    job = create_job(config)
    # define TabularDataset object to use in running CustomTrainingJob
    dataset_path = 'projects/'+config['project_id']+'/locations/'+config['region']+'/datasets/'+config['dataset_id']
    ds = aiplatform.TabularDataset(dataset_path)
    # run the CustomTrainingJob object to get a trained model
    model = run_job(job, ds, model_args,config)
    print("deployment starting")
    # deploy model to a Vertex AI endpoint
    if config['deploy_model']:
        deploy_model(model,config)
    print("pipeline completed")
    # show time taken by script
    print("--- %s seconds ---" % (time.time() - start_time))


