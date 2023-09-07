# ML pipeline for a Trained Deep Learning Model

This repo contains the code for an ML pipeline to train a deep learning model with a tabular dataset and deploy the trained model to a Vertex AI endpoint. This example is the core of a chapter for the upcoming Manning book on machine learning with tabular data.

File descriptions 

- [model_training_keras_preprocessing.py](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/model_training_keras_preprocessing.py): model training script - adapted from the notebook version of the training code: [model_training_keras_preprocessing.ipynb](https://github.com/ryanmark1867/deep_learning_best_practices/blob/master/notebooks/model_training_keras_preprocessing.ipynb)
- [model_training_config.yml](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/model_training_config.yml): config file for model_training_keras_preprocessing.py. This file is not accessed directly. Instead, it is copied to Google Cloud Storage and the URI for that blob is passed as an argument to the training code running in a container.
- [pipeline_script.py](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/pipeline_script.py): script for Kubeflow pipeline that invokes model_training_keras_preprocessing.py in a pre-built Vertex AI container
- [pipeline_config.yml](https://github.com/ryanmark1867/deep_learning_ml_pipeline/blob/master/pipeline_config.yml): config file containing parameters for pipeline_script.py

Here are the articles that describe the deployments in more detail:

- [Training models in Vertex AI containers](https://markryan-69718.medium.com/training-a-custom-model-in-a-pre-built-vertex-ai-container-8a34e244db53?sk=5486fc72b783977e03f8215339d58596) - this article provides an overview of how to use the code in this repo to set up training of a custom model in a Vertex AI pre-built container.
- [Developing and Deploying a Machine Learning Model on Vertex AI using Python](https://towardsdatascience.com/developing-and-deploying-a-machine-learning-model-on-vertex-ai-using-python-865b535814f8) - ideas from this article and the accompanying repo were used in this code in this repo.

