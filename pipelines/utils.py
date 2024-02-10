import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessingStrategy
import src.col_definition as col_def

def get_data_for_test():
    try:
        path = r"C:\Users\okore\OneDrive\Desktop\MACHINE_LEARNING\Anomaly_Detection\data\Test.txt"
        df = pd.read_csv(path, header=None, names=col_def.columns)
        df = df.sample(n=150)
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df= data_cleaning.handle_data()
               
        #df.drop( ["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e