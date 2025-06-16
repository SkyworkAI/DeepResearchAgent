import contextlib
import os

import httpx
from dotenv import load_dotenv

load_dotenv(verbose=True)

PROXY_URL = os.getenv("LOCAL_PROXY_BASE", None)

if PROXY_URL:
    TRANSPORT = httpx.HTTPTransport(proxy=httpx.Proxy(url=PROXY_URL))
    HTTP_CLIENT = httpx.Client(transport=TRANSPORT)
    ASYNC_TRANSPORT = httpx.AsyncHTTPTransport(proxy=httpx.Proxy(url=PROXY_URL))
    ASYNC_HTTP_CLIENT = httpx.AsyncClient(transport=ASYNC_TRANSPORT)
else:
    HTTP_CLIENT = httpx.Client()
    ASYNC_HTTP_CLIENT = httpx.AsyncClient()


@contextlib.contextmanager
def proxy_env(proxy_url: str = PROXY_URL):
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    try:
        yield
    finally:
        del os.environ["HTTP_PROXY"]
        del os.environ["HTTPS_PROXY"]


__all__ = [
    "PROXY_URL",
    "HTTP_CLIENT",
    "ASYNC_HTTP_CLIENT",
]
