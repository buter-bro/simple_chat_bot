from fastapi import APIRouter, WebSocket
from fastapi.responses import StreamingResponse
from starlette.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel

from configs.generation_config import generation_cfg
from model.generate import Generate
import json
from utils.enums import InferenceType

router = APIRouter()
token_generator = Generate(generation_cfg)


class InputText(BaseModel):
    input_text: str
    temperature: float
    eps: float


@router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_text()
    input_data = json.loads(data)
    inference_prefs = {
        'temperature': input_data['temperature'],
        'eps': input_data['eps']
    }
    generator = token_generator.token_generator(input_data['prompt'], inference_prefs)
    async for text in generator:
        await websocket.send_text(text)
    await websocket.send_text("[EOS]")


@router.post('/generate_sentence')
async def generate_endpoint(input_data: InputText):
    inference_prefs = {
        'temperature': input_data.temperature,
        'eps': input_data.eps
    }
    output_text = token_generator.generate_sequence(input_data.input_text, inference_prefs=inference_prefs)
    return PlainTextResponse(output_text)


