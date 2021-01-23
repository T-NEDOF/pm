
import psycopg2
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, text
from psycopg2 import pool
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

from app.main.config import config

def connect_postgres():
    return psycopg2.connect(host=config.POSTGRES_HOST, database=config.POSTGRES_DATABASE, user=config.POSTGRES_USER, password=config.POSTGRES_PASSWORD)

def execute_query(query):
    connection = connect_postgres()
    dfPostgres = pd.read_sql(query, connection)
    return dfPostgres

def execute_insert(query, data):
    try:
        connection = connect_postgres()
        cursor = connection.cursor()
        cursor.execute(query, data)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into table")
        return True
    except (Exception, psycopg2.Error) as error:
        if(connection):
            print("Failed to insert record into table", error)
        return False
    finally:
        # closing database connection.
        if(connection):
            cursor.close()
            connection.close()
        print("PostgreSQL connection is closed")
        return False
    
def execute_update(query, data):
    try:
        connection = connect_postgres()
        cursor = connection.cursor()
        cursor.execute(query, data)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record updated successfully in table")
        return True
    except (Exception, psycopg2.Error) as error:
        if(connection):
            print("Failed to update record in table", error)
        return False
    finally:
        # closing database connection.
        if(connection):
            cursor.close()
            connection.close()
        print("PostgreSQL connection is closed")
        return False
