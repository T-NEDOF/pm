from flask import Flask, jsonify, request
from flask_restplus import Resource

import pandas as pd
import os
import datetime
import requests
import time

from werkzeug.utils import secure_filename
from app.main.util.dto import SimulationDto

from app.main.service import db_service
from app.main.service import thingsboard_service

from app.main.config import config as cf
from packages.neural_network_model.neural_network_model.config import config

api = SimulationDto.api

@api.route('/status')
class Simulation(Resource):
    @api.response(201, 'Get simulation status successfully')
    @api.doc('Stop Simulation data')
    def get(self):
        # Get current simulation status in thingsboard database
        status = thingsboard_service.get_simulation_status()
        return {
            "success": True, 
            "message": "Get simulation status successfully", 
            "data": status }

@api.route('/start')
class SimulationStart(Resource):
    @api.response(201, 'Simulate data successfully')
    @api.doc('Simulation data')
    def get(self):  
        # Update simuation status in thingsboard database to 1 - Running simulation
        ts = datetime.datetime.now().timestamp()
        status_data = (1, ts)
        thingsboard_service.update_simulation_status(status_data)

        # Load sensor data from CSV
        df_train = pd.read_csv(f'{config.DATASET_DIR}/{config.LTSM_DATA_FILE}')
        X_train = df_train

        # List tokens of devices on thingsboard
        list_token = cf.LIST_TOKEN
        df_columns = cf.DEVICE_DATA_COLUMNS
        
        for i in range(len(X_train)):
            # Check simulation current status to keep simulating data or not
            simulation_status = thingsboard_service.get_simulation_status()
            if simulation_status == 0 and i > 0:
                break
            
            # Transform telemetry data from csv to thingsboard database
            data_telemetry = {}
            
            for j in range(len(df_columns)):
                data_telemetry['%s' % (df_columns[j])] = X_train['%s' % (df_columns[j])].iloc[i]
            data = '%s' % (data_telemetry)
            x = X_train['id'].iloc[i]

            # Call thingsboard service to simulate data into thingsboard system
            url_telemetry = cf.THINGSBOARD_URL + '/api/v1/%s/telemetry' % list_token[x-1]
            headers = {'Content-Type': 'application/json'}
            requests.post(url_telemetry, data=data, headers=headers)
            time.sleep(10)
        return {"success": True, "message": "Start simulation successfully"}

@api.route('/stop')
class SimulationStop(Resource):
    @api.response(201, 'Stop simulation successfully')
    @api.doc('Stop Simulation data')
    def get(self):
        # Update simuation status in thingsboard database to 0 - Stopped simulation
        ts = datetime.datetime.now().timestamp()
        status_data = (0, ts)
        thingsboard_service.update_simulation_status(status_data)
        return {"success": True, 
                "message": "Stop simulation successfully",
                "data": status_data}

@api.route('/initialize_data')
class SimulationInitialize(Resource):
    @api.response(201, 'Initialize sensor data successfully')
    @api.doc('Innitialize sensor data in postgres database')
    def get(self):  
        # Load simulating data for LSTM prediction
        df_train = pd.read_csv(f'{config.DATASET_DIR}/{config.LTSM_DATA_FILE}')
        
        # Only get data of devices 1, 2, 3, 4, 5 for simulation
        X_train = df_train[df_train.id < 6]
        X_train.reset_index(drop=True)

        # Sensor data columns in thingsboard database
        df_columns = cf.DEVICE_DATA_COLUMNS

        # Initialize data into thingsboard database
        for i in range(len(X_train)):
            data_telemetry = {}
            for j in range(len(df_columns)):
                data_telemetry['%s' % (df_columns[j])] = X_train['%s' % (df_columns[j])].iloc[i]
            data = data_telemetry.values()
            data = list(data)
            ts = datetime.datetime.now().timestamp()
            data.append(ts)
            data = tuple(data)
            thingsboard_service.insert_sensor_data(data)
            time.sleep(1)
        return {"success": True, "message": "Initialize sensor data successfully"}