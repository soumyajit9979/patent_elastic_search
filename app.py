import uvicorn

from server.utils import config

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        lifespan="on",
        workers=1,
        reload=bool(config.debug),
    )
