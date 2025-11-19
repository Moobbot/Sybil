import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from call_model import load_model
from config import API_CONFIG, FOLDERS, HOST_CONNECT, IS_DEV, PORT_CONNECT, SECURITY_CONFIG
from utils import cleanup_old_results, get_local_ip

logger = logging.getLogger("sybil.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load heavy resources once at startup."""
    global model

    cleanup_old_results([FOLDERS["CLEANUP"]])

    if model is None:
        logger.info("Loading Sybil model on startup...")
        model = load_model()

    import routes

    routes.model = model

    yield


model = None

app = FastAPI(
    title=API_CONFIG["TITLE"],
    description=API_CONFIG["DESCRIPTION"],
    version=API_CONFIG["VERSION"],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SECURITY_CONFIG["CORS_ORIGINS"],
    allow_credentials=True,
    allow_methods=SECURITY_CONFIG["CORS_METHODS"],
    allow_headers=SECURITY_CONFIG["CORS_HEADERS"],
)

app.mount("/results", StaticFiles(directory=FOLDERS["RESULTS"]), name="results")

from routes import router

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(("localhost", port)) == 0

    custom_port = PORT_CONNECT
    while is_port_in_use(custom_port):
        custom_port += 1

    local_ip = get_local_ip()
    print(f"Running on: http://127.0.0.1:{custom_port} (localhost)")
    print(f"Running on: http://{local_ip}:{custom_port} (local network)")

    uvicorn.run("api:app", host=HOST_CONNECT, port=custom_port, reload=IS_DEV)
