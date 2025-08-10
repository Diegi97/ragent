from logging.config import dictConfig

from colorlog import ColoredFormatter


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a colorlog formatter once per process."""
    fmt = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s"
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # keep any previously‑defined children
            "formatters": {
                "color": {
                    "()": ColoredFormatter,
                    "format": fmt,
                    "log_colors": {
                        "DEBUG": "white",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "bold_red",
                    },
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "color",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
            # ← NEW
            "loggers": {
                "httpx": {
                    "level": "WARNING",  # DEBUG/INFO are now muted
                    "propagate": True,  # still bubble up to root so colorlog prints it
                }
            },
        }
    )