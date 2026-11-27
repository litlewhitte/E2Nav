import requests
import base64
import json
import os
from PIL import Image, ImageDraw, ImageFont

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:72b"
IMAGE_PATH = "temp/demo.jpeg"
PROMPT = "Describe the contents of this image in detail."
# ----------------

def create_placeholder_image(path: str, text: str = "Hello, Qwen-VL!"):
    """Create a placeholder image when the input is missing."""
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
        
    img = Image.new('RGB', (400, 200), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((50, 80), text, fill=(255, 255, 0), font=font)
    img.save(path)
    print(f"Created a placeholder because image '{path}' was missing.")

def image_file_to_base64(filepath: str) -> str:
    """Encode an image as base64 for Ollama."""
    with open(filepath, "rb") as image_file:
        binary_data = image_file.read()
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return base64_string

def ask_qwen_vl(prompt: str, image_path: str):
    """Submit an image and text prompt to Qwen-VL through Ollama."""
    if not os.path.exists(image_path):
        create_placeholder_image(image_path)

    print(f"Processing image: {image_path}")
    b64_image = image_file_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image]
            }
        ],
        "stream": False
    }

    print("Sending request to Ollama...")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() 

        response_data = response.json()
        
        answer = response_data.get("message", {}).get("content")
        
        print("\n--- Qwen-VL response ---")
        if answer:
            print(answer)
        else:
            print("No answer found in response:")
            print(response_data)

        print("\n--- Statistics ---")
        print(f"Total time: {response_data.get('total_duration', 'N/A') / 1e9:.2f} seconds")
        print(f"Prompt tokens evaluated: {response_data.get('prompt_eval_count', 'N/A')}")
        print(f"Generated tokens evaluated: {response_data.get('eval_count', 'N/A')}")
        
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        print("Check that Ollama is running at http://localhost:11434.")
    except json.JSONDecodeError:
        print("\nInvalid JSON response:")
        print(response.text)

if __name__ == "__main__":
    ask_qwen_vl(PROMPT, IMAGE_PATH)
