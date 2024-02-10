from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    # Run the pipeline
    print("CLient URI: ",  Client().active_stack.experiment_tracker.get_tracking_uri())
    path = r"C:\Users\okore\OneDrive\Desktop\MACHINE_LEARNING\Anomaly_Detection\data\Train.txt"
    train_pipeline(data_path=path)
    
mlflow ui --backend-store-uri str(Client().active_stack.experiment_tracker.get_tracking_uri())
    
