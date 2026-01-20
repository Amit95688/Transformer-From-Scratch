"""
Airflow DAG for Transformer Training Pipeline
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

# Project root and logs
PROJECT_ROOT = "/app"  # Mount point in Docker
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Default DAG args
default_args = {
    'owner': 'ml_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False
}

# DAG definition
dag = DAG(
    'transformer_training_pipeline',
    default_args=default_args,
    description='DVC Transformer Training Pipeline: Ingestion → Transformation → Training',
    schedule='@weekly',
    catchup=False,
    tags=['ml', 'transformer', 'dvc']
)

# -------------------------------
# Stage 1: Data Ingestion
# -------------------------------
data_ingestion_task = BashOperator(
    task_id='data_ingestion',
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s data_ingestion >> {LOGS_DIR}/data_ingestion.log 2>&1",
    dag=dag
)

# -------------------------------
# Stage 2: Data Transformation
# -------------------------------
data_transformation_task = BashOperator(
    task_id='data_transformation',
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s data_transformation >> {LOGS_DIR}/data_transformation.log 2>&1",
    dag=dag
)

# -------------------------------
# Stage 3: Model Training
# -------------------------------
model_training_task = BashOperator(
    task_id='model_training',
    bash_command=f"cd {PROJECT_ROOT} && dvc repro -s model_training >> {LOGS_DIR}/model_training.log 2>&1",
    dag=dag
)

# -------------------------------
# Stage 4: Optional DVC Push
# -------------------------------
push_artifacts_task = BashOperator(
    task_id='push_artifacts',
    bash_command=f"cd {PROJECT_ROOT} && dvc push >> {LOGS_DIR}/dvc_push.log 2>&1",
    dag=dag
)

# -------------------------------
# Set execution order
# -------------------------------
data_ingestion_task >> data_transformation_task >> model_training_task >> push_artifacts_task