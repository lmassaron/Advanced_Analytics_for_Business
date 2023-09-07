# example of using Flask to access a Vertex AI deployment of a Keras deep learning model trained on a tabular dataset
import json
import os
import urllib.request
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf

# import libraries required for endpoint access
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# carry over imports from custom layer version
import time
# import datetime, timedelta
import datetime
from datetime import datetime, timedelta
from datetime import date
from dateutil import relativedelta
from io import StringIO
import pandas as pd
import pickle
from pickle import dump
from pickle import load
# DSX code to import uploaded documents
from io import StringIO
import requests
import json


import os
import yaml
import math
from flask import Flask, render_template, request

import tensorflow as tf
print(tf.__version__)

# load config gile
current_path = os.getcwd()
print("current directory is: "+current_path)
path_to_yaml = os.path.join(current_path, 'flask_web_deploy_config.yml')
print("path_to_yaml "+path_to_yaml)
try:
    with open (path_to_yaml, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file')
    
print("TF VERSION IS: ",tf.__version__)

# build the path for the trained model
rawpath = os.getcwd()
# models are in a directory called "models" in the same directory as this module
model_path = os.path.abspath(os.path.join(rawpath, 'models'))

print("path is:",rawpath)
print("model_path is: ",model_path)


# function from https://github.com/googleapis/python-aiplatform/blob/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py
def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if type(instances) == list else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    print("prediction is: ",predictions)
    return(predictions)


app = Flask(__name__)


@app.route('/')
def home():   
    ''' render home.html - page that is served at localhost that allows user to enter model scoring parameters'''
    title_text = "web deployment of Keras model"
    title = {'titlename':title_text}
    return render_template('home.html',title=title) 
    
@app.route('/show-prediction/')
def show_prediction():
    ''' 
    get the scoring parameters entered in home.html and render show-prediction.html
    '''
    # the scoring parameters are sent to this page as parameters on the URL link from home.html
    # load the scoring parameter values into a dataframe
    # create and load scoring parameters dictionary (containing the scoring parameters)that will be fed into the pipelines
    scoring_dict = {}
    for col in config['scoring_columns_cat']:
        print("value for "+col+" is: "+str(request.args.get(col))) 
        scoring_dict[col] = str(request.args.get(col))
    for col in config['scoring_columns_cont']:
        scoring_dict[col] = float(request.args.get(col))
    # hardcode size_type_bin for now
    scoring_dict['size_type_bin'] = str(request.args.get('size_type'))+' 1'
    # endpoint deployment must have no extraneous features
    scoring_dict.pop('size_type')
    # print details about scoring parameters
    print("scoring_dict: ",scoring_dict)
    input_dict = {name: [value] for name, value in scoring_dict.items()}
    print("input_dict: ",input_dict)
    # call to the function to get a prediction from the Vertex AI endpoint
    predictions = predict_custom_trained_model_sample(
    project = config['endpoint']['project'],
    endpoint_id = config['endpoint']['endpoint_id'],
    location = config['endpoint']['location'],
    instances = input_dict)
    prob = tf.nn.sigmoid(predictions[0])

    print(
        "This property has a %.1f percent probability of "
        "having a price over the median." % (100 * prob)
    )

    if prob < 0.5:
        predict_string = "Prediction is: property has a price less than median"
    else:
        predict_string = "Prediction is: property has a price more than median"
    # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key':predict_string}
    # render the page that will show the prediction
    return(render_template('show-prediction.html',prediction=prediction))
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')