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

from app.main.config import config
from app.main.service.total_cost import totalcost
from app.main.service import db_service

def insert_sensor_data(data):
    query = """ INSERT INTO sensor (id ,cycle, setting1, setting2, setting3, s1, s2, s3,
                        s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
                        s15, s16, s17, s18, s19, s20, s21, ttf, ts) 
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    db_service.execute_insert(query, data)
    return True

def get_simulation_status():
    query = """SELECT status
            FROM public.data_simulation
            WHERE key = 'simulation'"""
    df_data = db_service.execute_query(query)
    status = int(df_data['status'].iloc[0])
    return status

def update_simulation_status(data):
    print(data)
    query = """ UPDATE public.data_simulation SET status = %s, updated_at = %s WHERE key = 'simulation'"""
    result = db_service.execute_update(query, data)
    return result

def get_sensor_data_last_50_cycles(device_id, current_cycle):
    query="""SELECT id, cycle, setting1, setting2, setting3, s1, s2,s3,
                    s4, s5,s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16,
                    s17, s18, s19, s20, s21,ttf
                FROM public.sensor where id= '%d' AND cycle <= '%d'
                ORDER BY ts DESC
                LIMIT 51"""%(device_id, current_cycle)
    data = db_service.execute_query(query) 
    return data


        