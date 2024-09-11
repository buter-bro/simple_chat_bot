from easydict import EasyDict


model_cfg = EasyDict()

model_cfg.decoder = EasyDict()
model_cfg.decoder.d_model = 512
model_cfg.decoder.heads_num = 16
model_cfg.decoder.layers_num = 4

model_cfg.decoder.d_ff = 2048
model_cfg.decoder.activation = 'GELU'
model_cfg.decoder.dropout_rate = 0.1

model_cfg_v1 = EasyDict()

model_cfg_v1.decoder = EasyDict()
model_cfg_v1.decoder.d_model = 512
model_cfg_v1.decoder.heads_num = 16
model_cfg_v1.decoder.layers_num = 6

model_cfg_v1.decoder.d_ff = 2048
model_cfg_v1.decoder.activation = 'GELU'
model_cfg_v1.decoder.dropout_rate = 0.1

