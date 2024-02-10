import logging

import pandas as pd
from zenml import step
from src.evaluation import AccuracyScore, Recall, PrecisionScore
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step
def evaluate_model(model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "acc_score"],
    Annotated[float, "recall_score"],
    Annotated[float, "prec_score"],
]:
    """
    Evaluate the model on the ingested data.
    
    Args:
        X_test: Testing data,
        y_test: Testing labels,
    Returns:
        None
    """
    try:
        prediction = model.predict(X_test)
        AccuracyScore_class = AccuracyScore()
        acc_score = AccuracyScore_class.calculate_score(y_test, prediction)

        Recall_class = Recall()
        recall_score = Recall_class.calculate_score(y_test, prediction)

        Prec_class = PrecisionScore()
        prec_score = Prec_class.calculate_score(y_test, prediction)
        
        experiment_name = 'First_Experiment'
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metric('acc_score', acc_score)
            mlflow.log_metric('recall_score', recall_score)
            mlflow.log_metric('prec_score', prec_score)

        return acc_score, recall_score, prec_score
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e