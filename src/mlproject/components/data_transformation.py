import os
from src.mlproject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import pandas as pd
from src.mlproject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def get_data_transformer_object(self):
        # data = pd.read_csv("artifacts/data_ingestion/cardata.csv")
        data = pd.read_csv(self.config.data_path)
        categorical_columns = data.select_dtypes(include=object).columns.tolist()
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            if column!='Car_Name':
               data[column] = label_encoder.fit_transform(data[column])

        # self.data=data   
        # print(self.data)   
        return data             


    def train_test_spliting(self):
        # data = pd.read_csv(self.config.data_path)
        # Label encoding
        # categorical_columns = data.select_dtypes(include=object).columns.tolist()
        # label_encoder = LabelEncoder()
        # for column in categorical_columns:
        #     if column!='Car_Name':
        #        data[column] = label_encoder.fit_transform(data[column])


        # Split the data into training and test sets. (0.75, 0.25) split.
        # train, test = train_test_split(data)
        train, test = train_test_split(self.get_data_transformer_object())

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)