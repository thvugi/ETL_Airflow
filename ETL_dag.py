from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from twitter_etl import run_twitter_etl
from amazon_etl import run_amazon_etl

default_args = {

    'owner' : 'airflow',
    'depends_on_past' : False,
    'start_date' : datetime(2023, 12, 8),
    'emails': ['airflow@example.com'],
    'email_on_failure' : False,
    'email_on_retry' : False,
    'retries' : 1,
    'retry_delay': timedelta(minutes=1)
}


dag = DAG(
    'twitter_dag',
    default_args = default_args,
    description = 'ETL code for running dags'
)

run_etl = PythonOperator(
    task_id = 'complete_etl_tasks',
    python_callable = {run_twitter_etl, run_amazon_etl,},
    
    dag = dag,
)

run_etl