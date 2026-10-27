from __future__ import annotations

import logging
from pathlib import Path

from cmapss_xai.paths import ensure_dir


def configure_logging(command: str, reports_dir: Path) -> logging.Logger:
    log_dir = ensure_dir(reports_dir / "logs")
    logger = logging.getLogger("cmapss_xai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_dir / f"{command}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        from rich.logging import RichHandler

        console_handler: logging.Handler = RichHandler(rich_tracebacks=True, markup=False)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    except Exception:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)
    return logger
