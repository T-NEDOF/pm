import os
import keras
import datetime
import requests
import numpy as np
import time
import keras.backend as K
import pandas as pd
import dill as pickle
import json

from flask import request, jsonify
from flask_restplus import Resource
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model, model_from_json

from app.main.util.dto import RegressionDto
from app.main.service import thingsboard_service
from app.main.service import db_service
import tensorflow as tf

# from app.main.controller.maintenance.rul_engine_model.predict import make_prediction_by_reg_random_forest, make_prediction_by_reg_decision_tree
# from app.main.controller.maintenance.sssapp.main.controller.maintenance.rul_engine_model import __version__ as _version

from packages.neural_network_model.neural_network_model.processing import preprocessors as pp
from packages.neural_network_model.neural_network_model.processing import data_management as dm
from packages.neural_network_model.neural_network_model.config import config
from packages.neural_network_model.neural_network_model.predict import make_prediction_lstm_classification




api = RegressionDto.api

@api.route('/lstm')
class LSTMRegression(Resource):
    @api.response(201, 'LSTM regression')
    @api.doc('LSTM regression')

    def r2_keras(y_true, y_pred):
                """Coefficient of Determination
                """
                SS_res =  K.sum(K.square( y_true - y_pred ))
                SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
                return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    def Get_data(self):
        print('9'*10)
        # Get data from thingsboard requests
        K.clear_session()
        json_data = request.get_json()
        device_id = int(json_data['id'])
        current_cycle = int(json_data['cycle'])
    
        # Get sensor data from last 50 cycles of device
        df_last_50_cycles_data = thingsboard_service.get_sensor_data_last_50_cycles(device_id, current_cycle) 
        test_df = pp.preprocessing_data(df_last_50_cycles_data)
        train_data = dm.load_dataset(file_name=config.TRAINING_DATA_FILE)
        train_data = pp.preprocessing_train_data(train_data)
        test_df = pp.minmax_normalization(train_data, test_df)
        x = pp.GenSequence()
        

        def r2_keras(y_true, y_pred):
            """Coefficient of Determination
            """
            SS_res =  K.sum(K.square( y_true - y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )

        # print(config.DATASET_DIR)
        # print(config.REGRESSION_MODEL_FILE_NAME)
        
        estimator = load_model(config.DATASET_DIR + "/" + config.REGRESSION_MODEL_FILE_NAME, custom_objects={'r2_keras': r2_keras})
        y_pred_regression = estimator.predict(test_df)
        predict_rul = int(y_pred_regression[0])
        
        model_class = load_model(config.DATASET_DIR + "/"+config.CLASSIFICATION_MODEL_FILE_NAME)
        y_pred_class = model_class.predict_classes(test_df)
        # y_pred_class = make_prediction_lstm_classification(df_data)

        result = 'No'
        if y_pred_class[0] == 1 :
            result = 'Yes'
        K.clear_session()
        return {
            "success": True,
            "message": "Predict remaining useful life successfully",
            "predict_RUL": predict_rul, 
            "risk": result
            }


