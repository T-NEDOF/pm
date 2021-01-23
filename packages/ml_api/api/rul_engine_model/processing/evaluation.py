import pandas as pd
import math 

from app.main.controller.maintenance.rul_engine_model.config import config
from app.main.controller.maintenance.rul_engine_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

def get_regression_metrics(model, actual, predicted):
    
    """Calculate main regression metrics.
    
    Args:
        model (str): The model name identifier
        actual (series): Contains the test label values
        predicted (series): Contains the predicted values
        
    Returns:
        dataframe: The combined metrics in single dataframe
    
    
    """
     # view predictions vs actual
    pred_dict = {
                'Actual' : actual,
                'Prediction' : predicted
            }
    # pred = pd.DataFrame.from_dict(pred_dict)
    pred = pd.DataFrame.from_dict(pred_dict)
    error = pd.DataFrame(pred.Prediction-pred.Actual)
    error.columns = ['residuals']
    df_error=pd.concat([pred,error],axis=1)
    # calculated cost function
    result=0
    evaluation=[]
    for i in range(len(df_error)):
        if df_error['residuals'][i]<0:
            calculated = math.exp((df_error['residuals'][i])/-13)-1
            evaluation.append(calculated)
            result += calculated       
        else:
            calculated = math.exp((df_error['residuals'][i])/10)-1
            evaluation.append(calculated)
            result += calculated   
    total_cost = result/len(df_error) 
    
    MAPE= np.mean(np.abs((actual - predicted) / actual)) * 100
    regr_metrics = {
                        'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,
                        'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),
                        'R^2' : metrics.r2_score(actual, predicted),
                        'Explained Variance' : metrics.explained_variance_score(actual, predicted),
                        'Mean absolute percentage error': MAPE,
                        'Total cost': total_cost
                   }

    #return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
    df_regr_metrics.columns = [model]
    return df_regr_metrics

def view_predictions_actual(model, actual, predicted):
    # view predictions vs actual
    pred_dict = {
                'Actual' : actual,
                'Prediction' : predicted
            }
    # pred = pd.DataFrame.from_dict(pred_dict)
    pred_actual = pd.DataFrame.from_dict(pred_dict).T
    return pred_actual