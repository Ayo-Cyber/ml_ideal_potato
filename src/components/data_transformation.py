import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocess_obj_file = os.path.join('artifacts', 'preprocess.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            num_columns = ['writing_score', 'reading_score']
            cat_columns = ['gender',
                           'race_ethinicity',
                           'parental_level_of_education',
                           'lunch',
                           'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='median')),
                    ("Scaler", StandardScaler())
                ]
            )
            logging.info("Standard Scaling Completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("OHE", OneHotEncoder()),
                    ("Scaler", StandardScaler())
                ]
            )

            logging.info("Categorical Encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)

           logging.info("Reading Train and Test Data")

           logging.info("Obtaining Preprocessing Object")

           preprocessing_obj = self.get_data_transformer()

           target_column_name = 'math_score'
           numerical_columns = ['writing_score', 'reading_score']

           input_feature_train = train_df.drop(columns=[target_column_name], axis=1)
           target_feature_train_df = train_df[target_column_name]

           input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
           target_feature_test_df = test_df[target_column_name]

           
        except:
            pass


if __name__ == "__main__":


