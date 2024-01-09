import os
import sys
sys.path.append("C:/Users/ayven/Desktop/END_2_ENDML_PROJECT1/src")

from src.exception import CustomException
from src.logger import logging


import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df= pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the data set as dataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok =True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train_test_split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=102)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion of the data completed")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException (e, sys)


if __name__== "__main__":
    obj= DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
