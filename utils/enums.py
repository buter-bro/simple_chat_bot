from enum import IntEnum


class SetType(IntEnum):
    train = 1
    validation = 2
    test = 3


class WeightsInitType(IntEnum):
    normal = 1
    uniform = 2
    xavier_uniform = 3
    xavier_normal = 4
    kaiming_uniform = 5
    kaiming_normal = 6


class InferenceType(IntEnum):
    greedy = 1
    temperature = 2


class InferenceMode(IntEnum):
    token = 1
    sentence = 2