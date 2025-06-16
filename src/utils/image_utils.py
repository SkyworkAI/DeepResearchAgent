import base64
import mimetypes
import os
import uuid

import requests


def download_image(image_url, download_path):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

    request_kwargs = {
        "headers": {"User-Agent": user_agent},
        "stream": True,
    }

    # Send a HTTP request to the URL
    response = requests.get(image_url, **request_kwargs)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")

    extension = mimetypes.guess_extension(content_type)
    if extension is None:
        extension = ".download"

    fname = str(uuid.uuid4()) + extension
    download_image_path = os.path.join(download_path, fname)

    with open(download_image_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=512):
            fh.write(chunk)

    return download_image_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
