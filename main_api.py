import uvicorn
from fastapi import FastAPI, WebSocket

# from api.api_utils.logging import LogMiddleware
from api.inference_router import router as inference_router

app = FastAPI()

app.include_router(inference_router)
# app.add_middleware(LogMiddleware)


if __name__ == '__main__':
    uvicorn.run(app, host=f"0.0.0.0", port=8005)
