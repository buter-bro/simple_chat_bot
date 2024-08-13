import os
import threading
from typing import List

import clickhouse_connect
from dotenv import load_dotenv

from configs.inference_config import inference_cfg
from database.entities import *


class ClickHouse:
    _lock = threading.Lock()

    def __init__(self, to_create_tables=True):
        load_dotenv(inference_cfg.env_path)
        conn_params = {
            'host': os.environ['CLICKHOUSE_HOST'],
        }
        self.client = clickhouse_connect.get_client(**conn_params)
        self.database = os.environ['CLICKHOUSE_DATABASE']
        self.create_database_if_not_exists()
        self.client.command(f'USE {self.database}')

        if to_create_tables:
            self.create_tables()

    def create_database_if_not_exists(self):
        self.client.command(f'CREATE DATABASE IF NOT EXISTS `{self.database}`')


    def escape_sql_string(self, value):
        """Escapes unwanted symbols in value string."""
        return value.replace("'", "''").replace(";", "").replace("--", "")


    def create_tables(self):
        classes = Entity.get_concrete_classes()

        for class_table in classes:
            create_table_query = self.escape_sql_string(class_table.get_str_for_creating())
            engine = self.escape_sql_string(class_table._engine())
            after_engine = self.escape_sql_string(class_table._after_engine())
            self.client.command(
                f'CREATE TABLE IF NOT EXISTS `{class_table.__name__}` {create_table_query} ENGINE {engine} {after_engine}'
            )

    def insert(self, table: str, columns: List, values: List[List]):
        self.client.insert(f'{table}', values, column_names=columns)

    def select(self, sql_command: str, params: dict = {}):
        with self._lock:
            result = self.client.query_df(sql_command, params)
            return result