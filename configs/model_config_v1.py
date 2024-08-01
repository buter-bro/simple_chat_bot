from easydict import EasyDict


model_cfg = EasyDict()

model_cfg.decoder = EasyDict()
model_cfg.decoder.d_model = 512
model_cfg.decoder.heads_num = 16
model_cfg.decoder.layers_num = 8

model_cfg.decoder.d_ff = 2048
model_cfg.decoder.activation = 'GELU'
model_cfg.decoder.dropout_rate = 0.1

