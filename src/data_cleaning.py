import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as numpy
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import src.col_definition as col_def


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""

        try:
            imputer = SimpleImputer(strategy = 'mean')
            imputer.fit(data[col_def.numeric_cols])
            data[col_def.numeric_cols] = imputer.transform(data[col_def.numeric_cols])
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
        
              
        try:
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler.fit(data[col_def.numeric_cols])
            data[col_def.numeric_cols] = scaler.transform(data[col_def.numeric_cols])
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
        
        try:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(data[col_def.categorical_cols])
            encoded_cols = list(encoder.get_feature_names_out(col_def.categorical_cols))
            if data.shape[0] > 200:
                encoded_data = encoder.transform(data[col_def.categorical_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)

                data = pd.concat([data[col_def.numeric_cols], encoded_df, data[col_def.target_cols]],axis=1)
                return data
            else:
                data[encoded_cols] = encoder.transform(data[col_def.categorical_cols])
                data = pd.concat([data[col_def.numeric_cols], data[encoded_cols], data[col_def.target_cols]],axis=1)
                return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """

    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Divide data into train and test"""

        try:
            X = data
            y = data[col_def.target_cols]
            
            #print(X.columns.tolist())
            #print(y.columns.tolist())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        

class DataCleaning:
    """
    Class for cleaning data which preprocesses the data and divides it into train and test datasets
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle Data
        """

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
        

# if __name__ == "__main__":
#     data = pd.read_csv('../data/Train.txt', header=None, names=col_def.columns)
#     data_cleaning = DataCleaning(data, DataPreprocessingStrategy())
#     data_cleaning.handle_data()