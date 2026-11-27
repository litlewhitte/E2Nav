import requests
import json
import argparse
import os
from pathlib import Path # For robust path to URI conversion
from typing import List, Dict, Any
# import urllib.parse # Not strictly needed as Path.as_uri() handles encoding

def local_path_to_file_uri(file_path):
    """Converts a local file path to a file:/// URI."""
    try:
        abs_path = Path(file_path).resolve()
        if not abs_path.exists():
            print(f"Error: Local image file not found at {abs_path}")
            return None
        # Path.as_uri() correctly handles spaces and special characters on most OS
        file_uri = abs_path.as_uri()
        return file_uri
    except Exception as e:
        print(f"Error converting path {file_path} to file URI: {e}")
        return None

def run_inference_client(
    server_url: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 128
):
    endpoint = f"{server_url.rstrip('/')}/infer"
    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "do_sample": False
    }
    # Preview payload without potentially large base64 strings
    preview_payload_messages = []
    for msg in payload["messages"]:
        preview_content = []
        if "content" in msg and isinstance(msg["content"], list):
            for item in msg["content"]:
                item_copy = item.copy()
                if "image" in item_copy and isinstance(item_copy["image"], str) and len(item_copy["image"]) > 100:
                    item_copy["image"] = item_copy["image"][:50] + "...[truncated]"
                preview_content.append(item_copy)
        preview_payload_messages.append({"role": msg.get("role"), "content_preview": preview_content})

    print(f"向 {endpoint} 发送请求 (payload preview): {json.dumps({'messages_preview': preview_payload_messages, 'max_new_tokens': max_new_tokens}, indent=2)}")

    try:
        response = requests.post(endpoint, json=payload, timeout=300) # Increased timeout for large models
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        result = response.json()
        print("\n服务器响应:")
        print(f"  生成的文本: {result.get('generated_text')}")
        return result.get('generated_text')
    except requests.exceptions.HTTPError as http_err:
        print(f"\nHTTP错误: {http_err}")
        try:
            print(f"服务器错误详情: {http_err.response.json()}")
        except json.JSONDecodeError:
            print(f"服务器错误详情 (非JSON): {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n请求错误: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Inference Client for Qwen2.5-VL")
    parser.add_argument(
        "--server_url", type=str, default="http://localhost:8000",
        help="URL of the VLM inference server"
    )
    parser.add_argument(
        "--image_input", type=str, required=True,
        help="URL of the image or local path to the image file"
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image in detail.",
        help="Text prompt for the VLM"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Maximum new tokens to generate"
    )
    args = parser.parse_args()

    image_source_for_server = args.image_input

    # Check if the input is a local path and convert to file URI
    if not (args.image_input.startswith("http://") or args.image_input.startswith("https://") or args.image_input.startswith("data:image")):
        print(f"Input '{args.image_input}' looks like a local file path. Attempting to convert to file:/// URI.")
        file_uri = local_path_to_file_uri(args.image_input)
        if file_uri:
            image_source_for_server = file_uri
            print(f"Successfully converted local path to file URI: {image_source_for_server}")
        else:
            print(f"Warning: Failed to convert local image path '{args.image_input}' to file URI or file not found.")
            print(f"Sending path '{image_source_for_server}' as is. Server's qwen_vl_utils must be able to handle raw local paths if this is not a URI.")
            # Optionally, exit if conversion fails and you only want to support URIs or validated local paths:
            # print("Exiting due to image path issue.")
            # exit(1)
    else:
        print(f"Input '{args.image_input}' looks like a URL or data URI.")

    client_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source_for_server},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    run_inference_client(args.server_url, client_messages, args.max_new_tokens)
