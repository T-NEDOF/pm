import os
import pathlib

import app.main.controller.maintenance.rul_engine_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(app.main.controller.maintenance.rul_engine_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'

# target output
TARGET_TTF = 'ttf'
TARGET_BNC = 'label_bnc'

# original features
FEATURES_ORIG = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

# original + extracted fetures
FEATURES_ADXF = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9', 'av10', 'av11', 'av12', 'av13', 'av14', 'av15', 'av16', 'av17', 'av18', 'av19', 'av20', 'av21', 'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9', 'sd10', 'sd11', 'sd12', 'sd13', 'sd14', 'sd15', 'sd16', 'sd17', 'sd18', 'sd19', 'sd20', 'sd21']

# features with low or no correlation with regression label
FEATURES_LOWCR = ['setting3', 's1', 's10', 's18','s19','s16','s5', 'setting1', 'setting2']

# features that have correlation with regression label
FEATURES_CORRL = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20','s21']
FEATURES_IMPOR = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20','s21','sd2', 'sd3', 'sd4', 'sd6', 'sd7', 'sd8', 'sd9', 'sd11', 'sd12', 'sd13', 'sd14', 'sd15', 'sd17','sd20', 'sd21']


# Pipeline name of regresion model
# random forest
REG_RF_PIPELINE_NAME = 'reg_random_forest'
REG_RF_PIPELINE_SAVE_FILE = f'{REG_RF_PIPELINE_NAME}_output_v'

#decision tree
REG_DTR_PIPELINE_NAME = 'reg_decision_tree'
REG_DTR_PIPELINE_SAVE_FILE = f'{REG_DTR_PIPELINE_NAME}_output_v'

#linear regression
REG_LINEAR_REGRESSION_PIPELINE_NAME = 'reg_linear'
REG_LINEAR_REGRESSION_PIPELINE_SAVE_FILE = f'{REG_LINEAR_REGRESSION_PIPELINE_NAME}_output_v'

# Pipeline name of classification model
# random forest
CLF_RF_PIPELINE_NAME = 'clf_random_forest'
CLF_RF_PIPELINE_SAVE_FILE = f'{CLF_RF_PIPELINE_NAME}_output_v'

# decision tree
CLF_DTR_PIPELINE_NAME = 'clf_decision_tree'
CLF_DTR_PIPELINE_SAVE_FILE = f'{CLF_DTR_PIPELINE_NAME}_output_v'

# kneightbor
CLF_KNN_PIPELINE_NAME = 'clf_knn'
CLF_KNN_PIPELINE_SAVE_FILE = f'{CLF_KNN_PIPELINE_NAME}_output_v'

# svm
CLF_SVC_PIPELINE_NAME = 'clf_svc'
CLF_SVC_PIPELINE_SAVE_FILE = f'{CLF_SVC_PIPELINE_NAME}_output_v'

# GausionanNB
CLF_GAUSS_PIPELINE_NAME = 'clf_gauss'
CLF_GAUSS_PIPELINE_SAVE_FILE = f'{CLF_GAUSS_PIPELINE_NAME}_output_v'

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
