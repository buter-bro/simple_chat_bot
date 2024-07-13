from easydict import EasyDict
from configs.data_config import data_cfg
from configs.model_config import model_cfg


experiment_cfg = EasyDict()
experiment_cfg.data_cfg = data_cfg.tinystories_dataset
experiment_cfg.model = model_cfg.decoder

experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 16

experiment_cfg.validation = EasyDict()
experiment_cfg.validation.batch_size = 16

