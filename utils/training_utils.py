import math


def cosine_annealing_with_warmup(cur_step, t_max, warmup_steps=500, eta_min=0):
    if cur_step <= warmup_steps:
        return cur_step / warmup_steps
    else:
        return eta_min + (1 - eta_min) * (1 + math.cos(math.pi * (cur_step - warmup_steps) / (t_max - warmup_steps))) / 2
