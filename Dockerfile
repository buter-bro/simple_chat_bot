FROM python:3.9

WORKDIR /etc/simple_chat_bot/code

COPY ./api/ /etc/simple_chat_bot/code/simple_chat_bot/api
COPY ./model/ /etc/simple_chat_bot/code/simple_chat_bot/model
COPY ./main_api.py /etc/simple_chat_bot/code/simple_chat_bot/main_api.py
COPY ./utils/ /etc/simple_chat_bot/code/simple_chat_bot/utils
COPY ./configs/ /etc/simple_chat_bot/code/simple_chat_bot/configs
COPY ./requirements.txt /etc/simple_chat_bot/code/simple_chat_bot/requirements.txt

# костыль, переписать
COPY ./export/checkpoint_512371 /etc/simple_chat_bot/code/simple_chat_bot/experiments/no_f_ln_version/checkpoint_512371
COPY ./export/tokenization.pickle /etc/simple_chat_bot/code/simple_chat_bot/data/tinystories_dataset_v1/tokenization.pickle

ENV PYTHONPATH "${PYTHONPATH}:/etc/simple_chat_bot/code/simple_chat_bot"

RUN pip install --no-cache-dir -r /etc/simple_chat_bot/code/simple_chat_bot/requirements.txt

WORKDIR /etc/simple_chat_bot/code/simple_chat_bot/utils/tokenizer_utils
RUN python setup.py build_ext --inplace
WORKDIR /etc/simple_chat_bot/code

CMD ["uvicorn", "simple_chat_bot.main_api:app", "--host", "0.0.0.0", "--port", "8005"]
