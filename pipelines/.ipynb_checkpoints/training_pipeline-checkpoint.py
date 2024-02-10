from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model
import logging

@pipeline
def train_pipeline(data_path: str) -> None:
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    acc_score, recall_score, prec_score = evaluate_model(model, X_test, y_test)
    logging.info("Model Accuracy Score: {}".format(acc_score))