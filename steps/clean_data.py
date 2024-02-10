import logging
import pandas as pd  
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple
import src.col_definition as col_def

import src.col_definition as col_def

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
 ]:
    """
    Cleans the data and divides it into train and test
    
    Args:
        data: Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels"""
    try:
        #data = pd.read_csv('../data/Train.txt', header=None, names=col_def.columns)
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        process_data =  data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        logging.info('Data Cleaning Completed')
        print(X_train.drop(*col_def.target_cols, axis=1))
        print(y_train[col_def.target_cols])

        return X_train.drop(*col_def.target_cols, axis=1), X_test.drop(*col_def.target_cols, axis=1), y_train[col_def.target_cols], y_test[col_def.target_cols]
    except Exception as e:
        logging.exception("Error in dividing data: {}".format(e))
        raise e