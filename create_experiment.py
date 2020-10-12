import mlflow
from mlflow.tracking import MlflowClient
def create_experiment(experiment_name) -> int:
    """
    Creates an mlflow experiment
    :param experiment_name: str. The name of the experiment to be set in MLFlow
    :return: the id of the experiment created if it doesn't exist, or the id of the existing experiment if it is already
    :return: the id of the experiment created if it doesn't exist, or the id of the existing experiment if it is already
    created
    """
    client = MlflowClient()
    experiments = client.list_experiments()
    experiment_names = set(map(lambda e: e.name, experiments))
    if experiment_name in experiment_names:
        print(f'Experiment {experiment_name} already created.')
        return list(filter(lambda e: e.name == experiment_name, experiments))[0].experiment_id
    else:
        return mlflow.create_experiment(name=experiment_name)
