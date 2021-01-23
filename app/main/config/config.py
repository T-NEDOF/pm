# coding: utf-8
import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import os
import sys
import psycopg2
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, text
from psycopg2 import pool
import json
import requests
import numpy as np
import math
import datetime
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")
LOG_DIR = PACKAGE_ROOT / 'logs'
# LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'
UPLOAD_FOLDER = PACKAGE_ROOT / 'uploads'
# UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Thingsboard
THINGSBOARD_URL = "http://192.168.195.157:8080"

# Postgres production
POSTGRES_SCHEMA_NAME = "public" 
POSTGRES_HOST = "10.1.0.173"
POSTGRES_DATABASE = "thingsboard"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "12345678x@X"
POSTGRES_PORT = "5432"

# Simulated devices tokens - 5 devices with ids 1, 2, 3, 4, 5
# LIST_TOKEN = ["QjEh9A4QvmQBjvf9nICc", "lz8m1TBSapa4XpKPNc3r",
#             "3FkPxxdzn1pZYlAPbMNM", "GeC0FDaCk9pb8s8TZluO", "bcrILpC5DDa65dCe3lT2"]

LIST_TOKEN = ["gc5XIC4M4LusFA7JG28X", "RodfIKjBm8T7M4ZveQep",
            "R3x4fKMIAflglH4pq9al", "tTqHvWFRRMrFhLuum6Yn", "pPPd8w1PkVXL8NGFLf15"]

# Data definition
DEVICE_DATA_COLUMNS = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
            's15', 's16', 's17', 's18', 's19', 's20', 's21', 'ttf']

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


# def get_file_handler():
#     file_handler = TimedRotatingFileHandler(
#         LOG_FILE, when='midnight')
#     file_handler.setFormatter(FORMATTER)
#     file_handler.setLevel(logging.WARNING)
#     return file_handler


# def get_logger(*, logger_name):
#     """Get logger with prepared handlers."""

#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.INFO)
#     logger.addHandler(get_console_handler())
#     logger.addHandler(get_file_handler())
#     logger.propagate = False

#     return logger


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'this-really-needs-to-be-changed'
    SERVER_PORT = 5000
    UPLOAD_FOLDER = UPLOAD_FOLDER


class ProductionConfig(Config):
    DEBUG = False
    SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS', '0.0.0.0')
    SERVER_PORT = os.environ.get('SERVER_PORT', '5000')


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


def totalcost(df_error):
    # calculated cost function
    result = 0
    for i in range(len(df_error)):
        if df_error['total_cost'].iloc[i] < 0:
            calculated = math.exp((df_error['total_cost'].iloc[i])/-13)-1
            result += calculated
        else:
            calculated = math.exp((df_error['total_cost'].iloc[i])/10)-1
            result += calculated
    Total_cost = result/len(df_error)
    return int(Total_cost)


# uncomment the line below for postgres database url from environment variable
# postgres_local_base = os.environ['DATABASE_URL']

basedir = os.path.abspath(os.path.dirname(__file__))


config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

key = Config.SECRET_KEY
