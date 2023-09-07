# Simple Web Deployment for a Trained Deep Learning Model

This repo contains the code for a simple web deployment for a trained deep learning model for the upcoming Manning book on machine learning with tabular data. This is part of the code for the end-to-end deep learning and MLOps chapter.

Here are the key files in this repo:

- [flask_web_deploy.py](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy.py) - the Flask server module that loads the model specified in the config file [flask_web_deploy_config.yml](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy_config.yml)  and contains view functions to drive the `home.html` and show-`prediction.html` pages. This version is 100% local and reads the stored model from the local filesystem.

- [flask_endpoint_deploy.py](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_endpoint_deploy.py) - the Flask server module that interacts with the model deployed in a Vertex AI endpoint and gets the input parameters and shows the model predictions in the same web pages as  [flask_web_deploy.py](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy.py). To use this deployment, you must update the endpoint parameters in [flask_web_deploy_config.yml](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy_config.yml) to match your own Vertex AI endpoint parameters. You can get these parameters by clicking on the Sample Request link in the Deploy & Test tab for your model version in the Vertex AI Model Registry.

- [flask_web_deploy_config.yml](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy_config.yml) - contains the parameters to control the action of the Flask server module

- [home.html](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/templates/home.html) main web page of the deployment where the user can enter details about the property for which they want to get a price prediction

- [show-prediction.html](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/templates/show-prediction.html) web page to show the result of the model's prediction on the property whose details were input in [home.html](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/templates/home.html)

- [main2.css](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/static/css/main2.css) CSS file to control the rendering of the HTML pages

- [models/kl_real_estate_keras_preprocessing_model](https://github.com/ryanmark1867/deep_learning_web_deployment/tree/master/models/kl_real_estate_keras_preprocessing_model) model that is served by this application, as specified in the config file [flask_web_deploy_config.yml](https://github.com/ryanmark1867/deep_learning_web_deployment/blob/master/flask_web_deploy_config.yml)

There is a [separate repo](https://github.com/ryanmark1867/deep_learning_best_practices) containing the code to [prep the data](https://github.com/ryanmark1867/deep_learning_best_practices/blob/master/notebooks/data_preparation.ipynb) and [train the model](https://github.com/ryanmark1867/deep_learning_best_practices/blob/master/notebooks/model_training_keras_preprocessing.ipynb).   



Here are the articles that describe the deployments in more detail:

- [Deployment with a  model in the local file system](https://markryan-69718.medium.com/a-better-way-to-deploy-keras-models-a9d5764de964?sk=7ee732fad6eb2310fb0094cc44bff4bd)
- [Deployment with a model served from a Vertex AI endpoint](https://markryan-69718.medium.com/deploy-keras-models-with-vertex-ai-32e5b6d59f4f?sk=f36754de23fccdb3fa8a06b43c1271a9)