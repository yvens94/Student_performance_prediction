import os
import sys



import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from SRC.utils import save_object

from SRC.exception import CustomException
from SRC.logger import logging


from dataclasses import dataclass
from SRC.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:  str = os.path.join('artifact', "preprocessor.pkl")
    
   
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
 

    def get_data_transformer_object(self):
        logging.info("Data transformation has started")
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]

            num_pipeline = Pipeline(
                steps =[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                    ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='infrequent_if_exist')),
                    #("scaler", StandardScaler())
                    ]
                
            )

            logging.info(f"numerical columns standard scaling completed:{numerical_columns}")

            logging.info(f"categorical columns encoding completed:{categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                    ]
                    
            )


            return preprocessor
        
        
        
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.info("read train and test completed")

            logging.info("Obtainning preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            numerical_columns= ["writing_score", "reading_score"]
            target_column_name = "math_score"
            

            input_feature_train_df = train_df.drop([target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]



            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            preprocessing_obj.fit(input_feature_train_df)
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
            
        
# if __name__=="__main__":
#     obj= DataTransformation()
#     obj.initiate_data_transformation()