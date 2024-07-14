import mlflow
from typing import Union, List


class MLFlowLogger:
    def __init__(self, config):
        self.config = config
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)


    def _init_experiment(self):
        mlflow.set_experiment(self.config.experiment_name or "default_name")
        mlflow.enable_system_metrics_logging()

        if self.config.run_id:
            mlflow.start_run(run_id=self.config.run_id, log_system_metrics=True)
        else:
            mlflow.start_run(log_system_metrics=True)

        mlflow.log_artifact(self.config.dependencies_path)

    def log_hyperparameters(self, params: dict):
        mlflow.log_params(params)

    def save_metrics(self, set_type, metric_name: Union[List[str], str], metric_value: Union[List[float], float], step):
        if isinstance(metric_name, List):
            for m, v in zip(metric_name, metric_value):
                mlflow.log_metric(f'{set_type}_{m}', v, step)
        else:
            mlflow.log_metric(f'{set_type}_{metric_name}', metric_value, step)

    def save_plot(self, type_set, plot_name, plt_fig):
        plot_path = f"temp_plots/{type_set}_{plot_name}.png"
        plt_fig.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    def stop(self):
        mlflow.end_run()


