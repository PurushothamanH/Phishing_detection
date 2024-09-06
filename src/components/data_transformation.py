import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # You can use any classifier
from scipy.sparse import csr_matrix, hstack
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_filepath = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self,input_feature_train_df,target_feature_train_df,input_feature_test_df,target_feature_test_df):
        """ Responsible for data transformation..
        """

        try:
            # Initialize VarianceThreshold to remove features with zero variance
            selector = VarianceThreshold(threshold=0)  # threshold=0 removes all zero-variance features
            # Apply to your training data
            x_train_reduced = selector.fit_transform(input_feature_train_df)
            selected_columns = input_feature_train_df.columns[selector.get_support()]
            x_train_reduced_df = pd.DataFrame(x_train_reduced, columns=selected_columns)
            x_test_reduced = selector.transform(input_feature_test_df)
            x_test_reduced_df = pd.DataFrame(x_test_reduced, columns=selected_columns)
            

            # Variables to store the best k and its score
            best_k = None
            best_score = -1  # Initialize with a low score

            # Define the pipeline with SelectKBest and a classifier
            for k in range(5, 20):  # Try different values for k
                pipeline = Pipeline([
                    ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
                    ('classifier', RandomForestClassifier())
                ])
                scores = cross_val_score(pipeline, x_train_reduced_df, target_feature_train_df, cv=5)  # 5-fold cross-validation
                mean_score = scores.mean()
                
                # print(f"k={k}, mean accuracy={mean_score:.4f}")
                
                # Check if the current mean score is the best
                # if mean_score > best_score:
                #     best_score = mean_score
                #     best_k = k

            # Print the best k and its score
            # print(f"\nBest k: {best_k}, with a mean accuracy of: {best_score:.4f}") == {k=19,score=0.97}

            # Perform feature selection based on the ANOVA F-test for classification
            selector = SelectKBest(score_func=f_classif, k=19)
            X_train_selected = selector.fit_transform(x_train_reduced_df,target_feature_train_df)
            X_test_selected = selector.transform(x_test_reduced_df)
            original_columns = x_train_reduced_df.columns
            selected_columns_mask = selector.get_support()
            selected_columns = original_columns[selected_columns_mask]

            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_columns)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_columns)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected_df)
            X_test_scaled = scaler.transform(X_test_selected_df)
            x_train_scaled_df = pd.DataFrame(X_train_scaled,columns=X_train_selected_df.columns)
            x_test_scaled_df = pd.DataFrame(X_test_scaled,columns=X_test_selected_df.columns)
            
            return x_train_scaled_df,x_test_scaled_df,scaler


        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Train and Test data completed..")
            logging.info("obtaining Preprocessing object..")

            target_column_name ="CLASS_LABEL"

            input_feature_train_df = train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_data[target_column_name]
            input_feature_test_df = test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_data[target_column_name]

           # Obtain the preprocessed and scaled training and testing dataframes
            x_train_scaled_df, x_test_scaled_df,preprocessor_obj =self.get_data_transformer_object (
                input_feature_train_df, target_feature_train_df, input_feature_test_df, target_feature_test_df
            )

            logging.info("Applied preprocessing object on both train and test data.")
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            input_feature_train_arr = np.array(x_train_scaled_df)
            input_feature_test_arr = np.array(x_test_scaled_df)

            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)
            
            logging.info(f"saved Preprocessing object...")
            # target_column_name ="CLASS_LABEL"

            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )

            
        except Exception as e:
            raise CustomException(e,sys)
            