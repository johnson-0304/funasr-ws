"""`python -m funasr_ws` starts the server with settings from the environment."""

import logging

import uvicorn

from .config import load_settings
from .server import create_app


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        ws_max_size=4 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
