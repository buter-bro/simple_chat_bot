from easydict import EasyDict
from configs.data_config import data_cfg
from configs.model_config import model_cfg, model_cfg_v1
from utils.enums import InferenceType
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

generation_cfg = EasyDict()
generation_cfg.data = data_cfg.tinystories_dataset
generation_cfg.model = model_cfg.decoder

generation_cfg.checkpoint_to_load = os.path.join(
    ROOT_DIR, 'experiments', 'no_f_ln_version', 'checkpoint_512371'
)

generation_cfg.inference = EasyDict()
generation_cfg.inference.type = InferenceType.greedy
generation_cfg.inference.temperature_value = 0.95
generation_cfg.inference.eps = 1e-9
generation_cfg.inference.stop_predict = 200
