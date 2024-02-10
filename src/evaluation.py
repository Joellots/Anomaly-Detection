import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation of models"""
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model:
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass


class AccuracyScore(Evaluation):
    """
    Evaluation Strategy that uses Accuracy Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy Score")
            acc = accuracy_score(y_true, y_pred)
            logging.info("Accuracy Score: {}".format(acc))
            return acc
        except Exception as e:
            logging.error("Error in calculating Accuracy Score: {}".format(e))
            raise e
        
class Recall(Evaluation):
    """
    Evaluation Strategy that uses Recall Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Recall Score")
            recall = recall_score(y_true, y_pred, average='weighted')
            logging.info("Recall Score: {}".format(recall))
            return recall
        except Exception as e:
            logging.error("Error in calculating Recall Score: {}".format(e))
            raise e
        

class PrecisionScore(Evaluation):
    """
    Evaluation Strategy that uses Precision Score Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Precision Score")
            prec = precision_score(y_true, y_pred, average='weighted')
            logging.info("Precision Score: {}".format(prec))
            return prec
        except Exception as e:
            logging.error("Error in calculating Precision Score: {}".format(e))
            raise e
        