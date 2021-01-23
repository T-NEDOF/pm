import logging

import pandas as pd
import numpy as np
from packages.neural_network_model.neural_network_model.processing.data_management import load_pipeline_keras
# from neural_network_model import __version__ as _version
from packages.neural_network_model.neural_network_model.processing import preprocessors as pp
from packages.neural_network_model.neural_network_model.processing import data_management as dm
# from app.main.controller.maintenance.rul_engine_model.processing.validation import validate_inputs
import typing as t
# from tensorflow.python.framework import ops
import tensorflow as tf
import keras
import keras.backend as K
# from neural_network_model.config import config
from packages.neural_network_model.neural_network_model.config import config

_logger = logging.getLogger(__name__)

LSTM_CLASSIFICATION_PIPELINE = dm.load_pipeline_keras()
# config_tf = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
global graph
graph = tf.get_default_graph()

def make_prediction_lstm_classification(test_df) :
   
    """Make a single prediction using the saved model pipeline.

        Args:
            image_name: Filename of the image to classify
            image_directory: Location of the image to classify

        Returns
            Dictionary with both raw predictions and readable values.
        """
    # data = pd.DataFrame(input_data)
    # validated_data = validate_inputs(input_data=data)
    try :
            # with session.as_default():
            #     with session.graph.as_default():
        with keras.backend.get_session().graph.as_default():
            # tf.keras.backend.clear_session()
            test_df = pp.preprocessing_data(test_df)

            train_data = dm.load_dataset(file_name=config.TRAINING_DATA_FILE)
            train_data = pp.preprocessing_train_data(train_data)

            test_df = pp.minmax_normalization(train_data, test_df)
            np.savetxt('test_df.csv',test_df,delimiter=",")
            x = pp.GenSequence()
            test_df = x.transform(test_df)

            prediction = LSTM_CLASSIFICATION_PIPELINE.predict(test_df)
            print("#########################")
            print(prediction)
            # output = np.exp(prediction)
            results =  int(prediction[0])
            _logger.info(
                # f'Making predictions with model version: {_version} '
                # f'Inputs: {validated_data} '
                f'Predictions: {results}')
            # tf.keras.backend.clear_session()
            return results 
    except Exception as e:
        print(e)