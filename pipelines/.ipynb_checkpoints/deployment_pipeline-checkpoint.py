import json
import numpy as np
import pandas as pd
import logging
#from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
# MLFlowModelDeployer,
# )
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

from pipelines.utils import get_data_for_test
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model

import src.col_definition as col_def

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment Trigger Config"""
    min_accuracy: float = 0.92
    
@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Implements a model trigger to check if model is accurate enough for deployment"""
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters (BaseParameters):
    """MLflow deployment getter parameters
    
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
    ) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.
        Args:
            pipeline_name: name of the pipeline that deployed the MLflow prediction
                server
            step_name: the name of the step that deployed the MLflow prediction
                server
            running: when this flag is set, the step only returns a running service
            model_name: the name of the model that is deployed
    """
    # get the MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    
    # Configure the deployer to use the existing service
    # mlflow_model_deployer_component.config = {
    #     "service_path": " http://127.0.0.1:8080",  # Set this to the URL of your existing service
    #     # Additional configuration settings as needed
    # }

    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}."
            f"pipeline for the '{model_name}' model is currently "
            f"running."
)
    return existing_services[0]

@step
def predictor(
    service:MLFlowDeploymentService,
    data: str,
) -> np.ndarray:    
     # should be a NOP if already started
    #data = json.dump(data)
    data = json.loads(data)
    
    # data.pop("columns")
    # data.pop("index")
    #columns_for_df = []
    
    df = pd.DataFrame (data["data"], columns=data['columns'], index=data['index'])
    test_data= df.drop(*col_def.target_cols, axis=1)
    print(test_data)
    service.start(timeout=10)
    prediction = service.predict(test_data)
    return prediction

 
@pipeline(enable_cache=False, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92, 
    workers: int = 1, 
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    acc_score, recall_score, prec_score = evaluate_model(model, X_test, y_test)
    logging.info("Model Accuracy Score: {}".format(acc_score))
    deployment_decision = deployment_trigger(acc_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
    

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline (pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction
        
