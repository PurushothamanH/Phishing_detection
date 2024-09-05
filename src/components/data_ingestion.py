import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationconfig
# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

@dataclass
## this dataclass used to intialize the class variable inside class without using init directly...
### inputs are given to the below class for processing..
class DataIngestionconfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

### whenever this below function works it stores all data from above class like path of those three
### files in below function
class DataIngestion:
    def __init__(self):
        ### the reason to use init method instead of using dataclass is if only using or assigning a variable inside class then dataclass
        ### is ok. but assigning another func inside class then need to go to init method..
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the Data Ingestion method')
        try:
            df = pd.read_csv(r'notebook\data\Phishing.csv')
            logging.info('Read the dataset as dataframe')

            ## to create directory to store all those files..
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train Test Split Initiated..")
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion is completed..")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    # train_data,test_data=obj.initiate_data_ingestion()
    # data_transformation = DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    # # data_transformation.initiate_data_transformation(train_data,test_data)

    # try:
    #     modeltrainer = ModelTrainer()
    #     modeltrainer.initiate_model_trainer(train_arr,test_arr)
    # except Exception as e:
    #     raise CustomException(e,sys)



