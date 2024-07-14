from easydict import EasyDict
from configs.data_config import data_cfg
from configs.model_config import model_cfg


experiment_cfg = EasyDict()
experiment_cfg.data_cfg = data_cfg.tinystories_dataset
experiment_cfg.model = model_cfg.decoder

experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 16
experiment_cfg.train.lr = 0.001
experiment_cfg.train.weight_decay = 1e-2
experiment_cfg.train.label_smoothing = 0
experiment_cfg.train.warmup_steps = 1000
experiment_cfg.train.T_max = 10000
experiment_cfg.train.continue_train = False
experiment_cfg.train.epoches = 5

experiment_cfg.validation = EasyDict()
experiment_cfg.validation.batch_size = 16

# MlFlow config
experiment_cfg.mlflow = EasyDict()
experiment_cfg.mlflow.dependencies_path = 'requirements.txt'
experiment_cfg.mlflow.experiment_name = "simple_chat_bot"
experiment_cfg.mlflow.tracking_uri = None
experiment_cfg.mlflow.run_id = None

experiment_cfg.data = data_cfg.tinystories_dataset
experiment_cfg.model = model_cfg.decoder
