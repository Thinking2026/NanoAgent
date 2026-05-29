from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from demo.demo import Demo
from config import ConfigReader
from utils.env_util.env_loader import load_dotenv
from utils.log.log import Logger, zap
from utils.env_util.runtime_env import set_project_root


def _redirect_stdlib_logging(log_dir: Path) -> None:
    """Send all stdlib logging output to a file so it never pollutes the terminal."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "stdlib.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)


def main() -> None:
    project_root = set_project_root(_project_root)
    os.chdir(project_root)
    load_dotenv(project_root / ".env", override=True)
    # SSL_CERT_FILE must be applied before any HTTPS connection is made.
    # Python's ssl module reads this env var at context-creation time, so
    # setting it via load_dotenv (which runs after import) is too late unless
    # we explicitly rebuild the default context here.
    if ssl_cert := os.environ.get("SSL_CERT_FILE"):
        import ssl as _ssl
        _ssl.create_default_context  # noqa: ensure module loaded
        import urllib.request as _urllib_request
        _urllib_request.install_opener(
            _urllib_request.build_opener(
                _urllib_request.HTTPSHandler(
                    context=_ssl.create_default_context(cafile=ssl_cert)
                )
            )
        )
    config_path = project_root / "config" / "config.json"
    config = ConfigReader(config_path)
    log_dir = project_root / str(config.get("log.dir", "var/logs"))
    _redirect_stdlib_logging(log_dir)
    logger = Logger.get_instance(log_dir)
    try:
        application = Demo(config_path)
        application.run()
    except Exception as exc:
        logger.error(
            "Agent application initialization failed or run encountered an error",
            zap.any("config_path", config_path),
            zap.any("error", exc),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
