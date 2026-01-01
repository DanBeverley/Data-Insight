import os
import logging
from typing import Optional
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

logger = logging.getLogger(__name__)

config = Config(environ=os.environ)

oauth = OAuth()

GOOGLE_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "")

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile", "prompt": "select_account"},
    )
    logger.info("Google OAuth configured")
else:
    logger.warning("Google OAuth not configured - OAUTH_CLIENT_ID/SECRET missing")


def get_oauth() -> OAuth:
    return oauth


def is_google_oauth_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
