import boto3
import json
import mlflow
import mlflow.sagemaker as mfs
import mlflow.sklearn
from create_experiment import create_experiment
from eval_metrics import eval_metrics
from load_dataset import load_dataset
from sklearn.ensemble import RandomForestClassifier 
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info("create experiment ...")
    experiment_id = create_experiment('Flight delayed prediction')
    n_estimator= 100
    min_sample_split = 6
    rdste = 124
    logging.info("Start run experiment ...")

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        mlflow.set_tracking_uri("./mlruns")
        x_train, x_test, y_train, y_test = load_dataset(
        './data/488753924_T_ONTIME_REPORTING.csv', './data/2309682.csv',','
    )
        logging.info("Initialization of random Forest Model ...")
        rf = RandomForestClassifier(n_estimators = n_estimator, min_samples_split=min_sample_split, random_state=rdste)
        logging.info("Training ...")
        rf.fit(x_train, y_train)
        logging.info("Model Evaluation ...")
        f1score, balanced_accuracy = eval_metrics(y_test, rf.predict(x_test))
        print("F1-score ", f1score)
        print("Balanced accuracy ", balanced_accuracy)
        mlflow.log_metric("f1_score", f1score)
        mlflow.log_metric("balanced_accuracy_score", balanced_accuracy)
        mlflow.log_param("n_estimators", n_estimator)
        mlflow.log_param("min_samples_split", min_sample_split)
        logging.info("We finish by log model ...")

        mlflow.sklearn.log_model(rf, "model")
        
