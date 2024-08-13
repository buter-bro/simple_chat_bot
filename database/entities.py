from abc import ABC, abstractmethod
from enum import Enum


class AutoRegisterMeta(type(Enum), type(ABC)):
    def __new__(cls, name, bases, classdict):
        new_cls = type.__new__(cls, name, bases, classdict)
        for b in bases:
            if hasattr(b, 'register_subclass'):
                b.register_subclass(new_cls)
        return new_cls


class Entity(ABC, Enum, metaclass=AutoRegisterMeta):
    _subclasses = []

    @classmethod
    def register_subclass(cls, subclass):
        cls._subclasses.append(subclass)

    @classmethod
    def get_concrete_classes(cls):
        return cls._subclasses

    @classmethod
    def get_str_for_creating(cls):
        string_for_creating = '('
        for k, v in cls.__dict__.items():
            if not k.startswith("_"):
                string_for_creating += f'{k} {v},'
        string_for_creating = string_for_creating[:-1] + ')'
        return string_for_creating

    @staticmethod
    @abstractmethod
    def _after_engine():
        pass

    @staticmethod
    def _engine():
        return "MergeTree"


class ModelLogs(Entity):
    request_body = 'String'
    response_status = 'Int32'
    response_body = 'String'
    request_date = 'DateTime'

    @staticmethod
    def _after_engine():
        return 'ORDER BY request_date'


class GenerateLogs(Entity):
    input_text = 'String'
    generated_text = 'String'
    request_date = 'DateTime'

    @staticmethod
    def _after_engine():
        return 'ORDER BY request_date'