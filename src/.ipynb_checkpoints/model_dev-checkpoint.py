import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            
        Returns:
            None
        """

        pass

class LogisticRegressionModel(Model):
    """
    Logistic Regression Model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model using Logistic Regression
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            logreg = LogisticRegression(**kwargs)
            print(X_train.to_numpy())
            print(y_train.values.ravel())
            logreg.fit(X_train.to_numpy(), y_train.values.ravel())
            logging.info("Logistic Regression Model Training Completed")
            return logreg
        except Exception as e:
            logging.exception("Error in training logistic regression model: {}".format(e))
            raise e

class RandomForestModel(Model):
    """
    Random Forest Model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model using Logistic Regression
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            forest = RandomForestClassifier(**kwargs)
            forest.fit(X_train.to_numpy(), y_train.values.ravel())
            logging.info("Random Forest Model Training Completed")
            return forest
        except Exception as e:
            logging.exception("Error in training random forest model: {}".format(e))
            raise e
