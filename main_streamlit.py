import streamlit as st
import requests
import websocket
import itertools
import re


def send_generation_request(endpoint, data):

    url = f"http://simple_chat_bot:8005/{endpoint}"
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.text
    except:
        return None


def response_generator(prompt, host='simple_chat_bot', port='8005'):
    ws = websocket.create_connection(f"ws://{host}:{port}/ws")
    ws.send(prompt)
    while True:
        response = ws.recv()
        if response == '[EOS]':
            break
        yield response + ' '
    ws.close()


st.title("Tiny stories chat bot")
st.caption("")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Start a simple story:"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(itertools.chain(re.split('( )', prompt + ' '), response_generator(prompt)))
    st.session_state.messages.append({"role": "assistant", "content": response})


    #
    # async def get_bot_response():
    #     response_placeholder = st.empty()
    #     bot_response = prompt + " "
    #     async for response_chunk in send_receive(prompt):
    #         bot_response += response_chunk + ' '
    #         response_placeholder.markdown(bot_response)
    #     response_placeholder.markdown(bot_response)
    #     st.session_state.messages.append({"role": "assistant", "content": bot_response})
    #
    # with st.chat_message("assistant"):
    #     asyncio.run(get_bot_response())



