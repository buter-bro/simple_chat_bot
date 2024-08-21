import streamlit as st
import requests
import websocket
import itertools
import re
import json


def send_generation_sequence_request(prompt, temperature, stop_predict, eps, host='simple_chat_bot', port='8005'):

    url = f"http://{host}:{port}/generate_sentence"
    data = {
        'input_text': prompt,
        'temperature':  temperature,
        'stop_predict': stop_predict,
        'eps': eps
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.text
    except:
        return None


def response_generator(prompt, temperature, stop_predict, eps, host='simple_chat_bot', port='8005'):

    data = {
        'prompt': prompt,
        'temperature': temperature,
        'stop_predict': stop_predict,
        'eps': eps
    }
    data_to_load = json.dumps(data)
    ws = websocket.create_connection(f"ws://{host}:{port}/ws")
    ws.send(data_to_load)
    while True:
        response = ws.recv()
        if response == '[EOS]':
            break
        yield response + ' '
    ws.close()


st.title("Tiny stories chat bot")
st.caption("")

temperature = st.slider("Select temperature", 0.0, 1.0, value=0.9, step=0.1)

stop_predict = st.slider("Select max sequence length in tokens", 50, 500, value=200, step=1)

generate_mode = st.selectbox(
    "Select generation mode",
    ("by token", "all at once"),
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Start a simple story:"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if generate_mode == 'by token':
            response = st.write_stream(
                itertools.chain(re.split('( )', prompt + ' '), response_generator(
                    prompt, temperature, stop_predict, 1e-9
                ))
            )
        elif generate_mode == 'all at once':
            response = send_generation_sequence_request(prompt, temperature, stop_predict, 1e-9)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

