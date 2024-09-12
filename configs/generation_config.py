from easydict import EasyDict
from configs.data_config import data_cfg
from configs.model_config import model_cfg, model_cfg_v1
from utils.enums import InferenceType
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

generation_cfg = EasyDict()
generation_cfg.tokenizer_name = 'YouTokenToMe'
generation_cfg.model_name = 'TransformerV1'
generation_cfg.data = data_cfg.tinystories_dataset
generation_cfg.model = model_cfg_v1.decoder

generation_cfg.checkpoint_to_load = os.path.join(
    ROOT_DIR, 'experiments', 'f_ln_version', 'checkpoint_61200'
)

generation_cfg.inference = EasyDict()
generation_cfg.inference.temperature_value = 0.95
generation_cfg.inference.eps = 1e-9
generation_cfg.inference.stop_predict = 200
