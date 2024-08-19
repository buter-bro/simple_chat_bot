from fastapi import APIRouter, WebSocket
from fastapi.responses import StreamingResponse
from starlette.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel

from configs.generation_config import generation_cfg
from model.generate import Generate

router = APIRouter()
token_generator = Generate(generation_cfg)


class InputText(BaseModel):
    input_text: str


@router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_text()
    generator = token_generator.token_generator(message)
    async for text in generator:
        await websocket.send_text(text)
    await websocket.send_text("[EOS]")


@router.post('/generator')
def generate_endpoint(input_text: InputText):
    return StreamingResponse(token_generator.token_generator(input_text.input_text), media_type="text/event-stream")


@router.post('/generate')
async def generate_endpoint(input_text: InputText):
    output_text = token_generator.token_generator(input_text.input_text)
    return PlainTextResponse(output_text)


