from easydict import EasyDict
from configs.data_config import data_cfg
from configs.model_config import model_cfg, model_cfg_v1
from utils.enums import InferenceType
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_cfg = EasyDict()
experiment_cfg.data = data_cfg.tinystories_dataset
experiment_cfg.model = model_cfg.decoder
experiment_cfg.model_name = 'Transformer'

experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 16
experiment_cfg.train.lr = 0.0005
experiment_cfg.train.weight_decay = 0.1
experiment_cfg.train.label_smoothing = 0
experiment_cfg.train.warmup_steps = 1000
# experiment_cfg.train.T_max = 10000
experiment_cfg.train.continue_train = True
experiment_cfg.train.checkpoint_from_step = 681028
experiment_cfg.train.num_epoches = 5
experiment_cfg.train.validation_frequency = 4000
experiment_cfg.train.validation_interval = 400
experiment_cfg.train.log_frequency = 100
experiment_cfg.train.log_window = 100
experiment_cfg.train.accum_gradient_iter = 8

experiment_cfg.validation = EasyDict()
experiment_cfg.validation.batch_size = 16

# MlFlow config
experiment_cfg.mlflow = EasyDict()
experiment_cfg.mlflow.dependencies_path = 'requirements.txt'
experiment_cfg.mlflow.experiment_name = "simple_chat_bot"
experiment_cfg.mlflow.tracking_uri = None
experiment_cfg.mlflow.run_id = '9bd416801c2b4cb4be67dbe53859422c'

# Checkpoints parameters
# experiment_cfg.checkpoints_dir = os.path.join(
#     ROOT_DIR, 'experiments', 'no_f_ln_version'
# )
experiment_cfg.checkpoints_dir = os.path.join(
    ROOT_DIR, 'experiments', 'no_f_ln_version'
)
experiment_cfg.checkpoint_save_frequency = 400
experiment_cfg.checkpoint_files_count = 10
experiment_cfg.checkpoint_name = 'checkpoint_%s'
experiment_cfg.best_checkpoint_name = 'best_checkpoint'
experiment_cfg.checkpoint_to_load = os.path.join(
    ROOT_DIR, 'experiments', 'no_f_ln_version', 'best_checkpoint'
)

experiment_cfg.overfit = EasyDict()
experiment_cfg.overfit.num_iterations = 500

# Inference parameters
experiment_cfg.inference = EasyDict()
experiment_cfg.inference.type = InferenceType.greedy
experiment_cfg.inference.temperature_value = 0.95
experiment_cfg.inference.eps = 1e-9
experiment_cfg.inference.stop_predict = 200  # Maximum number of inference steps (i.e. generated sequence length)
