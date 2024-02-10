import logging
import pandas as pd 
from zenml import step 

from src.model_dev import LogisticRegressionModel, RandomForestModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
    ) -> ClassifierMixin:
    """
    Trains the model on the ingested data.
    
    Args:
        df: The ingested data
    """

    try:
        model = None
        if config.model_name == "LogisticRegression":
            #mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
            mlflow.sklearn.log_model(model, "model")
            trained_model = model.train(X_train, y_train)
            return trained_model
        if config.model_name == "RandomForest":
            model = RandomForestModel()
            mlflow.sklearn.log_model(model, "model")
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e