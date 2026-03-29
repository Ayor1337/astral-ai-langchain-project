from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app.core.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def auth_env():
    previous = {
        "JWT_SECRET_KEY": os.environ.get("JWT_SECRET_KEY"),
        "JWT_EXPIRE_SECONDS": os.environ.get("JWT_EXPIRE_SECONDS"),
        "JWT_ALGORITHM": os.environ.get("JWT_ALGORITHM"),
    }
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    os.environ["JWT_EXPIRE_SECONDS"] = "604800"
    os.environ["JWT_ALGORITHM"] = "HS256"
    get_settings.cache_clear()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
