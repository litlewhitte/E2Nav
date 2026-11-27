"""HTTP transport shared by text and vision-language inference."""

import requests


def post_chat(url, payload, timeout=60, authorization="", session=None):
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    sender = requests if session is None else session
    response = sender.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
