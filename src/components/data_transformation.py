import os 
import sys 
from dataclasses import dataclass

import numpy as np
import  pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from src.exception import CustomExecption
from src.logger import logging
from src.utlis import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_features=['reading_score', 'writing_score']
            cat_features=['race_ethnicity', 'parental_level_of_education']
            cat_features_2=['gender', 'lunch','test_preparation_course']

            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                     ("Standardscaler",StandardScaler())
                ]

            )
            cat_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                     ("OneHotEncoder",OneHotEncoder()),
                     ("Standardscaler",StandardScaler(with_mean=False))
                ]

            )
            ordinal_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("OrdinalEncoder",OrdinalEncoder())
                ]

            )
            logging.info("numerical columns standard caling completed")
            logging.info(f"Numerical columns:{numerical_features}")
            logging.info(f"Categorical columns:{cat_features}")
            logging.info(f"ordinal  columns:{cat_features_2}")
            preprocessor = ColumnTransformer([
                    ("num_pipeline", numerical_pipeline,numerical_features),
                    ("ordinal_pipeline", ordinal_pipeline, cat_features_2),
                     ("onehot_pipeline", cat_pipline,cat_features),
                     ])
            
            
            return preprocessor
            
        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomExecption(e, sys)
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data= pd.read_csv(test_path)
            logging.info("read train and test data completed") 


            logging.info("obtaining preprocessing object")

            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_columns_name='math_score'
            numerical_features=['reading_score', 'writing_score']
            cat_features=['race_ethnicity', 'parental_level_of_education']
            cat_features_2=['gender', 'lunch','test_preparation_course']
            input_feature_train_df=train_data.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df=train_data[target_columns_name]
            input_feature_test_df=test_data.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df=test_data[target_columns_name]
            logging.info('applying preprocessing object on training ddataframe and testng dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomExecption(e, sys)
            
                                            