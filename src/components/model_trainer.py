import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            ## again split the dataset into train test split...
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # Define the model
            model = Sequential()
            model.add(Dense(64, input_dim=19, activation='relu'))  # Input layer with 64 neurons
            model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons
            model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron (binary classification)
            
            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

            # Train the model
            history = model.fit(x_train,y_train, epochs=10, batch_size=32, validation_split=0.2)

            # Evaluate the model
            loss, accuracy = model.evaluate(x_test,y_test)
            print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

            logging.info("Model developed...")

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj = model
            )

        except Exception as e:
            raise CustomException(e,sys)