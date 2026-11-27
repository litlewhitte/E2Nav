from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import os
import torch
import random
import re

import requests

from utils.chat_client import post_chat as _post_ollama_chat

from requests.adapters import HTTPAdapter
import socket
import base64
import time
import signal
from functools import wraps

from io import BytesIO
import json



socket.setdefaulttimeout(60)

_vlm_session = requests.Session()
_vlm_adapter = HTTPAdapter(
    pool_connections=50,
    pool_maxsize=50,
    pool_block=False,
    max_retries=0
)
_vlm_session.mount('http://', _vlm_adapter)
_vlm_session.mount('https://', _vlm_adapter)

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def numpy_to_base64(img_np: np.ndarray) -> str:
    assert img_np.dtype == np.uint8
    assert img_np.ndim == 3 and img_np.shape[2] == 3
    image = Image.fromarray(img_np)
    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG") 
    byte_data = byte_stream.getvalue()
    encoded_image = base64.b64encode(byte_data).decode("utf-8")
    return encoded_image 


def save_numpy_array_to_temp_file_uri(
    rgb_array: np.ndarray,
    prefix: str = "temp_vlm_image_",
    suffix: str = ".jpg"
):
    pil_img = Image.fromarray(rgb_array.astype(np.uint8), 'RGB') # Ensure uint8 for PIL
    with tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix, mode='wb') as tmp_file:
        pil_img.save(tmp_file, format="JPEG")
        tmp_file_path_str = tmp_file.name # Get the path before closing

    abs_path = Path(tmp_file_path_str).resolve()
    file_uri = abs_path.as_uri()
    # print(f"Saved NumPy array to temporary file: {tmp_file_path_str}, URI: {file_uri}")
    return file_uri


def room_type(url, model, view_rgb, optional_room_list, curr_detection_nodes, authorization=""):
    if not curr_detection_nodes:
        return (None, None, "uncertain", "")

    indexed_objects_description = []
    for i, node in enumerate(curr_detection_nodes):
        indexed_objects_description.append(f"Index {i}: {node.class_name}")

    max_valid_index = len(curr_detection_nodes) - 1
    formatted_new_objects_list_with_indices = "\n".join(indexed_objects_description)

    prompt = f"""
# Role
You are a mobile robot tasked with identifying room types in an indoor environment based on visual input.

# Objective
Analyze the provided image to:
1. Determine the **room_type** of the area shown.
2. Identify the single **most decisive object** from a given list that confirms this room type.
3. Assess your **confidence level** in the room type identification.
4. Provide a **description** explaining your analysis and reasoning for the room type identification.

# Constraints and Guidelines
1.  **Room Type Selection:**
    *   You MUST select a `room_type` from the following list: {optional_room_list}.
    *   Outputting any room type NOT EXACTLY from this list is STRICTLY FORBIDDEN.
    *   If you are **absolutely certain** the room does NOT belong to any category in {optional_room_list}, output `None` for `room_type`.

2.  **Confidence Assessment (CRITICAL):**
    *   You MUST output a `confidence` field with one of these values: "high", "low", or "uncertain".
    *   **"high"**: Use ONLY when you are VERY CONFIDENT (≥90% certainty) that the room type is correct based on clear, unambiguous visual evidence (e.g., multiple distinctive objects, clear room layout, obvious room function). When confidence is "high", you MUST output a valid `room_type` from the list.
    *   **"low"**: Use when you have some evidence but are NOT highly confident (50-90% certainty) - there are some indicators but also ambiguity or missing context. When confidence is "low", you MAY output a `room_type` if you have a reasonable guess, but it is acceptable to output `null` for `room_type`.
    *   **"uncertain"**: Use when you are UNSURE or the evidence is insufficient (<50% certainty) - the image is ambiguous, lacks clear room-defining features, or could belong to multiple room types. **CRITICAL**: When confidence is "uncertain", you MUST output `null` for `room_type` (do NOT guess a room type).
    *   **IMPORTANT**: Only output "high" confidence when you are TRULY CERTAIN. When in doubt, choose "low" or "uncertain", and when "uncertain", always set `room_type` to `null`.

3.  **Decisive Object Selection:**
    *   You will be given a list of detected objects in the current image, each with a 0-based index.
    *   Your task is to select THE SINGLE most room-defining object from THIS LIST.
    *   The `object_id` you output MUST be the 0-based index of your chosen object from this list.
    *   **CRITICAL:** The `object_id` must be an integer between 0 and {max_valid_index} (inclusive), corresponding to an object in the `Detected Objects List` below.

# Detected Objects List (for `object_id` selection):
{formatted_new_objects_list_with_indices}

# Analysis Protocol
1.  **Room-Defining:** The object MUST be strongly indicative of the determined `room_type`.
2.  **From Provided List:** The object MUST be one of those explicitly listed in the `Detected Objects List` above.
3.  **Centrality (Tie-breaker):** If multiple objects are equally room-defining, prefer objects closer to the image center.
4.  **Lowest Index (Final Tie-breaker):** If still tied, choose the object with the lowest index.

4.  **Description (REQUIRED):**
    *   You MUST output a `description` field containing a brief analysis of the room.
    *   The description should explain: (1) what visual evidence you observed, (2) why you chose this room type (or why you are uncertain), (3) what key objects or features influenced your decision.
    *   Keep the description concise (2-4 sentences) but informative.
    *   Even when `room_type` is `null`, you MUST still provide a description explaining why you cannot determine the room type.

# CRITICAL: OUTPUT FORMAT REQUIREMENTS
1. Your entire response MUST be a single, raw JSON object.
2. The response MUST start with the character '{{' and end with the character '}}'.
3. ABSOLUTELY NO other text, explanations, or markdown formatting is allowed.
4. It is STRICTLY FORBIDDEN to wrap the JSON in markdown blocks like ```json ... ```. Any response starting with '`' will be considered a failure.
5. The `object_id` MUST be a valid integer index from the provided list.
6. The `confidence` field MUST be one of: "high", "low", or "uncertain".
7. The `description` field MUST be a string containing your analysis.

# Here is an example of a PERFECT response (high confidence):
{{
  "room_type": "kitchen",
  "object_id": 2,
  "confidence": "high",
  "description": "I can clearly see a refrigerator, stove, and kitchen cabinets in the image. These are definitive kitchen features that strongly indicate this is a kitchen. The layout and presence of cooking appliances make this identification highly certain."
}}

# Here is an example when uncertain (room_type MUST be null):
{{
  "room_type": null,
  "object_id": 1,
  "confidence": "uncertain",
  "description": "The image shows a generic space with furniture like a chair and table, but lacks distinctive room-defining features. The objects could belong to multiple room types (living room, dining room, or study). Without clear indicators, I cannot confidently determine the room type."
}}

# Here is an example when low confidence (room_type can be null or a guess):
{{
  "room_type": "living_room",
  "object_id": 0,
  "confidence": "low",
  "description": "I see a sofa and coffee table which suggest a living room, but the view is limited and there are no other distinctive features. The room could also be a family room or den. I'm making a tentative guess based on limited evidence."
}}

# Now, analyze the image and provide your response strictly in the required raw JSON format.
"""
    prompt = prompt.replace("{{", "{").replace("}}", "}")
    img_vlm = numpy_to_base64(view_rgb)

    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    try:
        response_payload = _post_ollama_chat(
            url,
            payload,
            timeout=60,
            authorization=authorization
        )
    except requests.Timeout as timeout_err:
        print(f"[room_type] LLM request timed out: {timeout_err}")
        return (None, None, "uncertain", "")
    except (requests.RequestException, ValueError) as req_err:
        print(f"[room_type] LLM request failed or invalid JSON: {req_err}")
        return (None, None, "uncertain", "")

    content = response_payload.get("message", {}).get("content", "")

    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)

        if not json_match:
            return (None, None, "uncertain", "")

        json_string = json_match.group(0)

        detection = json.loads(json_string)
        room = detection.get("room_type")
        object_id = detection.get("object_id")
        confidence = detection.get("confidence", "uncertain")
        description = detection.get("description", "")

        if object_id is None or not isinstance(object_id, int) or not (0 <= object_id <= max_valid_index):
            return (None, None, "uncertain", description if description else "")

        object_node = curr_detection_nodes[object_id]
        return room, object_node, confidence, description
    except json.JSONDecodeError as e:
        return (None, None, "uncertain", "")
    except (KeyError, IndexError) as e:
        return (None, None, "uncertain", "")

def stairs_confirmed(url, model, view_rgb, authorization=""):
    prompt = """
Your mission is to carefully evaluate the object **highlighted by the green bounding box** and determine if it is a set of **stairs**.

## Mandatory Analysis Process (Your internal thought process)
1.  **Focus on the Box**: First, meticulously analyze only the visual information *within* the green bounding box. Ignore everything outside of it.
2.  **Evidence for Stairs**: Look for definitive features of stairs. Key features include a series of distinct, level steps (treads), vertical surfaces between steps (risers), and potentially a handrail.
3.  **Make a Decision**: Based on the presence and clarity of these key features, decide if the object is stairs. If the defining features are present, the answer is `true`. If they are absent or the object is clearly something else (e.g., a ramp, a shelf), the answer is `false`.

## Output Requirement
You MUST respond ONLY with a single, valid JSON object. Do not include any text, explanations, markdown formatting (like ```json), or any other characters before or after the JSON block.

Your response must strictly follow this format:
{
"is_stairs": <boolean>,
"confidence": <float>,
"reason": <string>
}

- **"is_stairs"**: `true` if you identify clear features of stairs. `false` otherwise.
- **"confidence"**: Your confidence in the "is_stairs" decision (0.0 to 1.0). High confidence for clear views, lower for ambiguous or occluded ones.
- **"reason"**: A concise, one-sentence summary of the visual evidence. Example for true: "The object shows a clear series of ascending steps and risers." Example for false: "The object lacks any distinct steps, appearing as a sloped surface."
            """
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    response_payload = _post_ollama_chat(
        url,
        payload,
        authorization=authorization
    )
    content = response_payload.get("message", {}).get("content", "{}")
    results = json.loads(content)
    
    return results

@timeout(90)
def _object_confirmed_impl(url, model, view_rgb, object_type, similar_object, vocabulary, use_bbox=True, authorization=""):
    candidate_objects = list(set([object_type] + similar_object))
    random.shuffle(candidate_objects)
    candidate_objects_str = ", ".join(f'"{item}"' for item in candidate_objects)

    prompt = f"""
Your primary mission is to identify the object(s) **highlighted by the green bounding box** by choosing the best label from a given list. Your final goal is to help determine if the object is a "{object_type}".

## Crucial Instructions
1.  **Box is an Indicator**: The green box shows the *approximate location*. Identify the **entire object or objects** it points to.
2.  **Object vs. Non-Object First**: Before choosing from the list, you **MUST** first determine if the box highlights a distinct, three-dimensional object or just a static surface (like a floor, wall, ceiling) or an unrecognizable pattern. This is your most important first step.
3.  **Handling Multiple Objects**: The box may sometimes encompass **more than one** distinct object. If you see multiple instances of the *same* object type (e.g., two chairs side-by-side), your task is to identify them by their **common singular label** (e.g., choose `"chair"`, not `"chairs"` or `"sofa"`).

## Context
- **Proposed Label to Verify (for final output)**: "{object_type}"
- **Unbiased Candidate List (for your primary choice)**: {candidate_objects_str}
- **Special Label (for non-objects)**: `"background"`
- **Permitted Vocabulary (for all corrections)**: {vocabulary}

## Mandatory Reasoning Process: "Verify, Identify, Justify, then Conclude"

### Step 0: Verify if it's a Distinct Object
1.  Look closely at the area in the green box. Is it a self-contained object (or objects) with clear boundaries, depth, and form?
2.  Or is it a flat surface, a part of a larger architectural feature (wall, floor), or just a texture?
3.  If it is **NOT** a distinct object, your decision is made: you will choose the special label `"background"`. Skip to Step 2 to justify this.
4.  If it **IS** a distinct object (or objects), proceed to Step 1.

### Step 1: Analyze and Identify from the Unbiased List
1.  **(Assuming it is a distinct object from Step 0)** First, scan the entire area within the box. Determine if it contains a **single object** or **multiple distinct instances of the same object type**.
2.  Meticulously analyze the verifiable components of the object(s) (e.g., for a chair, look for legs, a seat, a backrest). Do not infer features that are not visually present.
3.  Now, look at the **Unbiased Candidate List**: {candidate_objects_str}.
4.  Choose the single best and most plausible label from THIS list to describe the object(s). **Crucially, if you saw multiple objects of the same type, choose their common singular label (e.g., for two chairs, choose 'chair').**
5.  State your choice clearly. For example: "My chosen best label is 'sofa'." or "My chosen best label is 'chair'." or "My chosen best label is 'background'."

### Step 2: Justify Your Choice
1.  Provide a concise, evidence-based reason for your choice from the previous steps.
2.  **If you chose a label from the candidate list**: Explain *why* you selected it based on its **verifiable parts**.
    - **Good Example (Single Object)**: "I chose 'sofa' because the object is a single, continuous piece of furniture with a long seat for multiple people, a backrest, and armrests, clearly distinguishing it from a 'chair'."
    - **Good Example (Multiple Objects)**: "I chose 'chair' because the bounding box contains two separate, distinct objects. Each object has the features of a chair (a seat, a backrest, legs). I did not choose 'sofa' because this is not one single, contiguous unit, but rather two individual chairs placed together."
3.  **If you chose `"background"`**: Explain *why* the area is not a distinct object.
    - **Good Example**: "I chose 'background' because the highlighted area is a flat, continuous surface consistent with the floor. It lacks any identifiable components of an object like legs or a distinct 3D structure."

### Step 3: Conclude for the Final JSON Output
1.  Now, consider your chosen best label from Step 1.
2.  If your chosen best label **is** `"{object_type}"`, you will set `object_present` to `true`.
3.  If your chosen best label **is NOT** `"{object_type}"` (this includes if you chose another object OR if you chose `"background"`), you will set `object_present` to `false`.
4.  If you set `object_present` to `false`, you MUST put your chosen best label into the `corrected_object_type` field. This corrected label must be a valid option from the full `Permitted Vocabulary`.
    - **Important**: If you chose `"background"`, then the `corrected_object_type` should be `"background"`.

## Output Requirement
You must respond ONLY with a single, valid JSON object, following the logic from Step 3. Do not include any text, explanations, or markdown formatting (like ```json) before or after the JSON block.

Your response must strictly follow this format:
{{
"object_present": boolean,  // `true` if your chosen best label from the candidate list matches "{object_type}". `false` otherwise.
"confidence": float,  // Your confidence in your choice from Step 1 (0.0 to 1.0).
"reasoning": string,  // Your justification from Step 2.
"corrected_object_type": string  // REQUIRED if and only if `object_present` is `false`. The value MUST be your chosen best label.
}}
Now, analyze the image, follow the "Verify, Identify, Justify, then Conclude" process, and provide your response as a single, raw JSON object.
"""
    
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    max_retries = 2
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response_data = _post_ollama_chat(
                url,
                payload,
                authorization=authorization,
                session=_vlm_session
            )
            content = response_data.get("message", {}).get("content", "")
            start = content.find('{')
            end = content.rfind('}')
            
            json_str = content[start:end+1]
            results = json.loads(json_str)
            return results
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    return {"object_present": False, "confidence": 0.0, "reasoning": "API call failed after retries", "corrected_object_type": "unknown"}

def object_confirmed(url, model, view_rgb, object_type, similar_object, vocabulary, use_bbox=True, authorization=""):
    try:
        return _object_confirmed_impl(url, model, view_rgb, object_type, similar_object, vocabulary, use_bbox, authorization)
    except TimeoutError:
        return {"object_present": False, "confidence": 0.0, "reasoning": "Function timeout", "corrected_object_type": "unknown"}




def object_detection(url, model, view_rgb, obj_goal, optional_room_dict, curr_detection_nodes, authorization=""):
    if curr_detection_nodes:
        detected_objects_str = "\n".join([f"- {node.class_name}" for node in curr_detection_nodes])
    else:
        detected_objects_str = "No objects were pre-detected in this view."
        
    room_scores_str = json.dumps(optional_room_dict)

    # --- Start of the New, More Robust Prompt ---
    prompt = f"""
# ROLE
You are a precision visual analysis AI for a robot. Your primary function is to follow instructions with extreme accuracy to determine if a target object is present. You MUST respond ONLY in JSON format.

# OBJECTIVE
Your goal is to determine the status of the target object, **"{obj_goal}"**, based on the image provided. You will follow a strict, prioritized process to distinguish between *direct visual confirmation* and *high-confidence inference*.

# INPUT DATA
1.  **Target Object**: "{obj_goal}"
2.  **Pre-Detected Objects**:
{detected_objects_str}
3.  **Room Score Guide**: {room_scores_str}
    (This guide provides a score for each room type based on its relevance to the target object. A score of 5 indicates the highest relevance.)

# STRICT DECISION PROTOCOL (Follow in order)

## PRIORITY 1: DIRECT VISUAL SEARCH
First, scan the entire image for the **"{obj_goal}"**.
- **IF YOU SEE THE OBJECT**: You MUST immediately conclude the search. Set `goal_object_found` to `true` and `is_likely_nearby` to `false`. Do not proceed to Priority 2.

## PRIORITY 2: HIGH-CONFIDENCE INFERENCE (Only if object is NOT seen)
- **IF YOU DO NOT SEE THE OBJECT**: You will now assess if it is likely nearby. Set `goal_object_found` to `false`. Then, you may set `is_likely_nearby` to `true` **ONLY IF AT LEAST ONE of these two conditions is met**:
    
    A. **High-Score Room Detected**: You identify the room in the image, and its score in the 'Room Score Guide' is **exactly 5**.
    
    B. **Critical Associated Object Detected**: You see an object that is almost always found with the "{obj_goal}" (e.g., seeing a 'toothbrush' when the goal is 'toothpaste'; seeing a 'pillow' when the goal is 'bed').

- **IF NEITHER OF THE ABOVE CONDITIONS IS MET**: You MUST set `is_likely_nearby` to `false`. It is STRICTLY FORBIDDEN to set `is_likely_nearby` to `true` based on vague possibilities or low-scoring rooms.

# REQUIRED OUTPUT
- Your ENTIRE response MUST be a single, valid JSON object.
- NO extra text, explanations, or markdown (like ```json) are allowed.
- The JSON structure MUST be EXACTLY as follows:

{{
"reasoning": {{
    "direct_visual_search_result": "<Describe the outcome of your visual search. 'Object found at...' or 'Object not found after a full scan.'>",
    "contextual_inference_analysis": "<ONLY IF a decision was made in Priority 2, explain here which condition (A or B) was met and why. Otherwise, state 'Not applicable as object was found' or 'No high-confidence indicators found'.>"
}},
"confidence": "<Your confidence level for the overall conclusion: 'High', 'Medium', or 'Low'>",
"goal_object_found": <boolean: true only if directly seen>,
"is_likely_nearby": <boolean: true only if an inference condition was met>
}}
            """
    # --- End of the New Prompt ---
    # print(prompt)
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response_payload = _post_ollama_chat(
            url,
            payload,
            authorization=authorization
        )
        content = response_payload.get("message", {}).get("content", "")
        # print("--- Raw VLM Output ---\n", content, "\n----------------------")
        # --- Robust JSON Parsing ---
        # 1. Clean potential markdown fences
        if content.strip().startswith("```json"):
            content = content.strip()[7:-3].strip()
        
        # 2. Parse the cleaned string
        results = json.loads(content)
        print("check obj mandatory detection results: ", results)
        # 3. Safely get values using .get() to avoid KeyError
        goal_found = results.get("goal_object_found")
        is_likely = results.get("is_likely_nearby")

        # 4. Validate that the values are booleans
        if not isinstance(goal_found, bool) or not isinstance(is_likely, bool):
            return None, None
            
        return goal_found, is_likely

    except json.JSONDecodeError as e:
        return None, None
    except Exception as e:
        return None, None

def nearby_confirmed(url, model, view_rgb, obj_goal, authorization=""):
    prompt = f"""
You are an expert AI assistant for a home robot. Your task is to analyze the provided image and determine the likelihood that a specific target object, **'{obj_goal}'**, can be found in or near the area shown. You must follow a strict chain of thought to ensure accuracy and provide your output in a structured JSON format.

**Target Object:** {obj_goal}

Follow these steps precisely in your reasoning:

**Step 1: Scene Analysis**
First, meticulously analyze the image.
- What type of room or area is depicted (e.g., kitchen, living room, bedroom, hallway, bathroom, office)?
- What are the key features, furniture, and objects visible that support your room identification? Provide a brief description.

**Step 2: Object Knowledge and Location Priors**
Second, access your general knowledge about the target object.
- Where is the object '{obj_goal}' most commonly found? List the typical room types or contexts.
- Consider all plausible locations. For example, a 'cushion' could be in a 'living room' or a 'bedroom'. For a generic object like a 'cabinet', acknowledge that it could appear in many different rooms.

**Step 3: Correlation and Logical Reasoning**
Third, correlate your scene analysis (Step 1) with your object knowledge (Step 2). This is the most critical step.
- Is the target object '{obj_goal}' directly and clearly visible in the image?
- If not visible, does the identified room type from Step 1 strongly match any of the typical locations for the object from Step 2?
- Based on this correlation, provide a detailed explanation. State clearly why the current view suggests a high or low probability of finding the object nearby. For instance, if you see a bathroom sink and the goal is 'toilet', the probability is very high. If you see a kitchen and the goal is 'bed', the probability is very low.

**Step 4: Probability Assignment**
Fourth, based on your reasoning in Step 3, assign a probability score between 0.0 and 1.0. This score represents your confidence that the '{obj_goal}' is in the current view or a directly accessible adjacent area. Adhere to these strict guidelines:
- **Score >= 0.9 (Very High Confidence):** Use this ONLY if the object is clearly visible, OR if the room is a definitive and primary location for the object (e.g., a 'toilet' in a clearly identifiable 'bathroom'; a 'stove' in a 'kitchen'). This is a strong positive signal.
- **0.5 <= Score < 0.9 (Moderate Confidence):** Use this if the room is a plausible but not exclusive location for the object (e.g., a 'cushion' in a 'living room', a 'chair' in a 'bedroom'). This indicates a reasonable possibility.
- **Score < 0.5 (Low Confidence):** Use this if the room type has a weak or no connection to the object (e.g., a 'sofa' in a 'kitchen'). Also, use this for generic objects that provide little information about their location (e.g., a 'cabinet' in a non-descript hallway), as a confident prediction is impossible. Your default stance must be cautious. Do not assign a high score without very strong evidence.

**Step 5: Final Output in JSON Format**
Finally, encapsulate your entire analysis into a single JSON object. Do not include any text outside of the JSON structure.

The JSON structure must be as follows:
{{
  "scene_analysis": {{
    "room_type": "string",
    "description": "string"
  }},
  "object_knowledge": {{
    "object_name": "{obj_goal}",
    "typical_locations": ["string", "..."]
  }},
  "reasoning": "string",
  "probability": float
}}
"""
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response_data = _post_ollama_chat(
            url,
            payload,
            authorization=authorization
        )
        content = response_data.get("message", {}).get("content", "")
        start = content.find('{')
        end = content.rfind('}')
        
        json_str = content[start:end+1]
        results = json.loads(json_str)
        print("object nearby check: ", results)
        if results.get("probability") >= 0.9:
            return True
        else:
            return False
    except Exception as e:
        return False
    
    



def object_comparison(url, model, view_rgb, challenge_object_types, authorization=""):    
    prompt = f"""You are an image analysis assistant. Your task is to determine the type of the primary object in the photo by selecting the MOST appropriate type from a GIVEN list.

                **Task Details:**

                1.  **Image Analysis:** Carefully examine the image to understand the visual characteristics of the primary object.
                2.  **Candidate Types:** You will be provided with a list of specific object types: {challenge_object_types}.
                3.  **Selection Criteria:**
                    *   **Accuracy:** The chosen type must accurately represent the object.
                    *   **Exclusivity & Handling No Fit:** You must select exactly ONE type from the provided list.
                        *   Your primary goal is to find the best fit from the list {challenge_object_types}.
                        *   However, if, after careful consideration, you determine that **NO type from the list {challenge_object_types} is a reasonably appropriate description** for the primary object in the image, then and only then set the `object_type` field in your JSON response to `null`.

                Please carefully analyze the image and respond strictly in the following JSON format. Do not include any text outside of this JSON structure:
                {{
                    "object_type": <string | null>, // The object type you choose from the list {challenge_object_types}. Set to `null` (JSON null, not the string "None") if no type from the list is appropriate.
                    "confidence": "<high | medium | low>",     // Your confidence in this assessment. If object_type is null, this reflects your confidence that none of the provided types are suitable (e.g., "high" if you are sure none fit, or "low" if you are uncertain).
                    "reasoning": "<string>"       // A brief explanation for your decision. If object_type is null, explain why none of the provided types from {challenge_object_types} were suitable for the object observed.
                }}
                """
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    response_payload = _post_ollama_chat(
        url,
        payload,
        authorization=authorization
    )
    
    content = response_payload.get("message", {}).get("content", "{}")
    results = json.loads(content)
    
    return results

def stairs_class(url, model, view_rgb, authorization=""):   
    # --- Start of the New, More Robust Prompt ---
    prompt = f"""
    # INSTRUCTIONS
    You are an expert AI assistant for an indoor robot. Your task is to analyze the area within the green bounding box in the provided image and classify the type of stairs.

    # STAIRS DEFINITIONS
    1.  **multi_floor_stairs**: These are standard stairs that connect different floors of a building (e.g., from the first floor to the second). They typically have a significant number of steps (usually more than 5-6 visible) and lead upwards to a different level or a landing that is clearly part of another story.
    2.  **single_level_steps**: These are a small set of steps (usually 1-4) that connect two slightly different elevations on the *same floor*. They do not lead to a new story. Examples include steps leading down into a sunken living room or up onto a small platform or stage.

    # ANALYSIS CRITERIA
    - Focus **exclusively** on the content inside the green bounding box.
    - Analyze visual cues:
        - **Number of Steps**: Are there many steps, or just a few?
        - **Angle/Destination**: Do the stairs appear to ascend steeply towards a new floor, or do they lead to an area that is visibly still part of the current room/floor?
        - **Context**: Does the surrounding area suggest a major level change (like leading to a second-floor hallway) or a minor one (like a raised dining area)?

    # REQUIRED OUTPUT
    - Your ENTIRE response MUST be a single, valid JSON object.
    - NO extra text, explanations, or markdown (like ```json) are allowed.
    - The JSON structure MUST be EXACTLY as follows:
    {{
    "is_stairs_in_bbox": boolean, // Is the object in the green box actually stairs?
    "stair_type": "string", // Must be one of: "multi_floor_stairs", "single_level_steps", or "not_applicable" (if is_stairs_in_bbox is false).
    "confidence": float, // Your confidence in the classification, from 0.0 to 1.0.
    "reasoning": "string" // A brief explanation of your decision based on visual evidence.
    }}
    """
    # --- End of the New Prompt ---
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    response_payload = _post_ollama_chat(
        url,
        payload,
        authorization=authorization
    )
    
    content = response_payload.get("message", {}).get("content", "{}")
    results = json.loads(content)
    
    return results


# def object_final_confirmed(url, model, view_rgbs, object_type, similar_object, vocabulary, object_belief=None, object_description=None):
#     if object_type not in similar_object:
#         similar_object.append(object_type)
#     if object_type == "table" and "sink" in similar_object:
#         del similar_object[similar_object.index("sink")]
#     random.shuffle(similar_object)
#     similar_object_str = ", ".join(f'"{s}"' for s in similar_object)
#     vocabulary_str = ", ".join(f'"{v}"' for v in vocabulary)

#     prompt = f"""
# You will receive {len(view_rgbs)} pictures of a target scene from multiple viewpoints.

# ## Your Mission: Holistic Object Identification

# Your goal is to identify a specific object **type** of interest. A **green bounding box** will highlight a key part of this object or group of objects to guide your attention.

# ## Critical Instruction: The Bounding Box is a FOCAL POINT, Not a Boundary

# The green box is an **anchor** or **focal point**. It is NOT a strict boundary for your analysis.
# - **Your primary task is to identify the fundamental object TYPE** represented by the highlighted content.
# - You MUST look both inside and outside the box to understand the full shape, structure, and function of the object(s).
# - Use the surrounding environment (e.g., is it in a kitchen, an office?) as crucial context for your decision.

# ## Critical Consideration: Handling Multiple Instances
# - **The bounding box may contain MORE THAN ONE instance of the same object type.**
# - **Example Scenario**: A single green box might encompass two chairs placed side-by-side. Your task is to identify the object type as "chair", NOT to misinterpret the combined shape as a single, larger object like a "sofa" or "bench".
# - **Your Focus**: Identify the repeating, fundamental unit. What is the core object being shown?

# ## Multi-View Analysis Instructions
# You are provided with {len(view_rgbs)} images from different viewpoints:
# 1. **Analyze ALL images**: Each viewpoint provides different information about the object's structure and features.
# 2. **Cross-view consistency**: Look for features that appear consistently across multiple views to confirm the object identity.
# 3. **Complementary information**: Use different angles to understand the complete 3D structure of the object.
# 4. **Best view selection**: If some views are clearer than others, prioritize the clearer views in your analysis.

# ## Contextual Information
# - **Candidate List (Your primary options)**: [{similar_object_str}]
# - **Permitted Vocabulary (For the final answer)**: [{vocabulary_str}]

# ## Evidence-Based Analysis (Your mandatory thought process)
# You must follow these steps in your reasoning:

# 1.  **Focal Point Analysis**: Briefly describe the visual features of the area **inside the green bounding box**. Note any repeating patterns or identical parts. (e.g., "two identical wooden chair backs," "a metallic leg," "a wooden surface with grain").

# 2.  **Holistic Object Inference**: Now, expand your view. Based on the focal point and the visual information around it, describe the **fundamental object type** present.
#     - **If multiple instances are present, focus on describing the characteristics of a single, representative instance.**
#     - For example: "The focal point shows two chairs. A single representative chair has a high back and four wooden legs. The combined silhouette might look wide, but it's clearly composed of two distinct 'chair' units."

# 3.  **Environmental Context**: Briefly note the object's location and surroundings. How does this context help confirm its identity? (e.g., "The objects are in a dining room, placed next to each other facing a table, which strongly supports the 'chair' identification.").

# 4.  **Comparative Evaluation (Crucial Step)**: Systematically evaluate the inferred object type against **EACH** category in the `Candidate List`. For every category:
#     -   **State the category** you are evaluating (e.g., "Evaluating against 'Sofa'").
#     -   **Compare Features**: Explain how the features of a **single representative object** align with or contradict this category. For the "two chairs vs. sofa" example, you would state: "The object is composed of two separate, identical units, each with its own set of legs and backrest. This structure contradicts the typical single-frame construction of a 'sofa'."
#     -   **Perform an "Open-Vocabulary Sanity Check"**: Ask yourself: "Without any lists, what is the most precise real-world term for this object?" Use this unconstrained knowledge to critique the fit of the category from the list. (e.g., "The most precise term is 'chairs' (plural), which points to 'chair' as the correct base type.")
#     -   **Provide a mini-conclusion** for that category (e.g., "Conclusion for 'Sofa': A poor fit. The visual evidence points to multiple smaller objects, not one large one.").

# 5.  **Synthesis and Final Decision**: After evaluating all candidates, you must:
#     -   **Declare your final choice**. This choice MUST be from the `Candidate List` AND also be present in the `Permitted Vocabulary`.
#     -   **Provide a Conclusive Justification**: Summarize why your chosen category is the single best fit. **Explicitly address the multiple-instance possibility if relevant**, explaining how you distinguished it from a single larger object. (e.g., "The final choice is 'chair' because the bounding box contains two distinct chair objects, not a single 'sofa'. This is evidenced by the clear separation, repeating structures, and individual leg sets.")

# ## Simulator Image Consideration
# These are simulator images. Prioritize the object's fundamental geometry, structure, and functional parts over simplified textures or unnatural lighting.

# ## Output Requirement
# - You must respond ONLY with a single, valid JSON object. Do not include any text or markdown formatting.
# - It is STRICTLY FORBIDDEN to wrap the JSON in markdown blocks like ```json ... ```.

# Your response must strictly follow this format:
# {{
#   "confirmed_object_type": "string",
#   "confidence_score": "float",
#   "reasoning": "string"
# }}
# """
#     prompt_lst = []
#     for view_rgb in view_rgbs:
#         img_vlm = numpy_to_base64(view_rgb)
#         prompt_lst.append({
#             "type": "image_url",
#             "image_url": {"url": img_vlm},
#         })
#     prompt_lst.append({"type": "text", "text": prompt})

#     headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": "You are a helpful room classification expert."},
#             {
#                 "role": "user",
#                 "content": prompt_lst,
#             },
#         ],
#         "response_format": {"type": "json_object"},
#         "stream": False,
#         "temperature": 0.0,
#         "top_p": 0.95
#     }
    
#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
#         content_json = response.json()
#         content = content_json["choices"][0]["message"]["content"]
        
#         cleaned_content = content.strip()
#         if cleaned_content.startswith('```json'):
#             cleaned_content = cleaned_content[7:]
#         if cleaned_content.endswith('```'):
#             cleaned_content = cleaned_content[:-3]
#         cleaned_content = cleaned_content.strip()
        
#         results = json.loads(cleaned_content)
        
#         print(f"[VLM Final] Qwen thinks this object is {results['confirmed_object_type']} in {similar_object}")
#         print(f"[VLM Final] Confidence: {results['confidence_score']}")
#         print(f"[VLM Final] Reasoning: {results['reasoning']}")
        
#         if results["confirmed_object_type"] == object_type:
#             return True
#         else:
#             return False
#     except Exception as e:
#         print(f"[VLM Final] Error: {e}")
#         return False



@timeout(600)
def _object_final_confirmed_impl(url, model, view_rgbs, object_type, similar_object, vocabulary, object_belief=None, object_description=None, authorization=""):
    if object_type not in similar_object:
        similar_object.append(object_type)
    if object_type == "table" and "sink" in similar_object:
        del similar_object[similar_object.index("sink")]
    random.shuffle(similar_object)
    similar_object_str = ", ".join(f'"{s}"' for s in similar_object)
    vocabulary_str = ", ".join(f'"{v}"' for v in vocabulary)

    prompt = f"""
You will receive a set of pictures of a target scene from multiple viewpoints.

## Your Mission: Holistic Object Identification

Your goal is to identify a specific object **type** of interest. A **green bounding box** will highlight a key part of this object or group of objects to guide your attention.

## Critical Instruction: The Bounding Box is a FOCAL POINT, Not a Boundary

The green box is an **anchor** or **focal point**. It is NOT a strict boundary for your analysis.
- **Your primary task is to identify the fundamental object TYPE** represented by the highlighted content.
- You MUST look both inside and outside the box to understand the full shape, structure, and function of the object(s).
- Use the surrounding environment (e.g., is it in a kitchen, an office?) as crucial context for your decision.

## Critical Consideration: Handling Multiple Instances
- **The bounding box may contain MORE THAN ONE instance of the same object type.**
- **Example Scenario**: A single green box might encompass two chairs placed side-by-side. Your task is to identify the object type as "chair", NOT to misinterpret the combined shape as a single, larger object like a "sofa" or "bench".
- **Your Focus**: Identify the repeating, fundamental unit. What is the core object being shown?

## Contextual Information
- **Candidate List (Your primary options)**: [{similar_object_str}]
- **Permitted Vocabulary (For the final answer)**: [{vocabulary_str}]

## Evidence-Based Analysis (Your mandatory thought process)
You must follow these steps in your reasoning:

1.  **Focal Point Analysis**: Briefly describe the visual features of the area **inside the green bounding box**. Note any repeating patterns or identical parts. (e.g., "two identical wooden chair backs," "a metallic leg," "a wooden surface with grain").

2.  **Holistic Object Inference**: Now, expand your view. Based on the focal point and the visual information around it, describe the **fundamental object type** present.
    - **If multiple instances are present, focus on describing the characteristics of a single, representative instance.**
    - For example: "The focal point shows two chairs. A single representative chair has a high back and four wooden legs. The combined silhouette might look wide, but it's clearly composed of two distinct 'chair' units."

3.  **Environmental Context**: Briefly note the object's location and surroundings. How does this context help confirm its identity? (e.g., "The objects are in a dining room, placed next to each other facing a table, which strongly supports the 'chair' identification.").

4.  **Comparative Evaluation (Crucial Step)**: Systematically evaluate the inferred object type against **EACH** category in the `Candidate List`. For every category:
    -   **State the category** you are evaluating (e.g., "Evaluating against 'Sofa'").
    -   **Compare Features**: Explain how the features of a **single representative object** align with or contradict this category. For the "two chairs vs. sofa" example, you would state: "The object is composed of two separate, identical units, each with its own set of legs and backrest. This structure contradicts the typical single-frame construction of a 'sofa'."
    -   **Perform an "Open-Vocabulary Sanity Check"**: Ask yourself: "Without any lists, what is the most precise real-world term for this object?" Use this unconstrained knowledge to critique the fit of the category from the list. (e.g., "The most precise term is 'chairs' (plural), which points to 'chair' as the correct base type.")
    -   **Provide a mini-conclusion** for that category (e.g., "Conclusion for 'Sofa': A poor fit. The visual evidence points to multiple smaller objects, not one large one.").

5.  **Synthesis and Final Decision**: After evaluating all candidates, you must:
    -   **Declare your final choice**. This choice MUST be from the `Candidate List` AND also be present in the `Permitted Vocabulary`.
    -   **Provide a Conclusive Justification**: Summarize why your chosen category is the single best fit. **Explicitly address the multiple-instance possibility if relevant**, explaining how you distinguished it from a single larger object. (e.g., "The final choice is 'chair' because the bounding box contains two distinct chair objects, not a single 'sofa'. This is evidenced by the clear separation, repeating structures, and individual leg sets.")

## Simulator Image Consideration
These are simulator images. Prioritize the object's fundamental geometry, structure, and functional parts over simplified textures or unnatural lighting.

## Output Requirement
- You must respond ONLY with a single, valid JSON object. Do not include any text or markdown formatting.
- It is STRICTLY FORBIDDEN to wrap the JSON in markdown blocks like ```json ... ```.

Your response must strictly follow this format:
{{
  "confirmed_object_type": "string",
  "confidence_score": "float",
  "reasoning": "string"
}}
"""
    image_payloads = [numpy_to_base64(view_rgb) for view_rgb in view_rgbs]
    messages = [
        {"role": "system", "content": "You are a helpful room classification expert."},
        {"role": "user", "content": prompt, "images": image_payloads}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    max_retries = 2
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            content_json = _post_ollama_chat(
                url,
                payload,
                authorization=authorization,
                session=_vlm_session
            )
            content = content_json.get("message", {}).get("content", "")
            results = json.loads(content)
            print("qwen think this object is ", results["confirmed_object_type"], " in ", similar_object, "and the reason is ", results["reasoning"])
            if results["confirmed_object_type"] == object_type:
                return True
            else:
                return False
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    return False

def object_final_confirmed(url, model, view_rgbs, object_type, similar_object, vocabulary, object_belief=None, object_description=None, authorization=""):
    try:
        return _object_final_confirmed_impl(url, model, view_rgbs, object_type, similar_object, vocabulary, object_belief, object_description, authorization)
    except TimeoutError:
        return False



# def object_final_confirmed(url, model, view_rgbs, object_type, similar_object, vocabulary, object_belief=None, object_description=None):
#     """
    
#     Args:
#     """
#     if object_type not in similar_object:
#         similar_object.append(object_type)
#     if object_type == "table" and "sink" in similar_object:
#         del similar_object[similar_object.index("sink")]
#     random.shuffle(similar_object)
#     similar_object_str = ", ".join(f'"{s}"' for s in similar_object)
#     vocabulary_str = ", ".join(f'"{v}"' for v in vocabulary)

#     stage1_prompt = f"""
# You are analyzing one or more images with a green bounding box highlighting a specific object. Your task is to provide a comprehensive scene analysis and object identification.

# ## Your Mission: Scene Analysis and Object Identification

# Please follow this chain of thought process:

# ### Step 1: Environment Analysis
# - What type of environment/room is this? (e.g., kitchen, bedroom, office, bathroom, living room, etc.)
# - Describe the overall setting and context

# ### Step 2: Inventory of Objects
# - What objects can you see in this environment?
# - List the main furniture, fixtures, and items present
# - Note their approximate locations and relationships

# ### Step 3: Green Box Object Analysis
# - Focus on the area highlighted by the green bounding box in the image(s)
# - What object(s) are contained within or near the green box?
# - Describe the object's visual features, materials, and characteristics
# - If multiple objects are present, identify the primary one
# - If multiple images are provided, analyze the object from different perspectives

# ### Step 4: Object Classification
# - Based on your analysis, what is the most likely object type in the green box?
# - Consider the object's function, appearance, and context

# ## Output Format
# Respond with a JSON object containing:

# {{
#     "environment": "description of the room/environment type",
#     "scene_objects": ["list", "of", "objects", "in", "the", "scene"],
#     "green_box_object": "description of the object in the green box",
#     "object_type": "your identified object type",
#     "reasoning": "brief explanation of your identification process"
# }}


# Focus on providing accurate, detailed analysis based on what you can actually see in the image.
# """

#     stage1_prompt_lst = []
    
#     for view_rgb in view_rgbs:
#         img_vlm = numpy_to_base64(view_rgb)
#         stage1_prompt_lst.append({
#             "type": "image_url",
#             "image_url": {"url": img_vlm},
#         })
    
#     additional_rgb = view_rgbs[0]
#     additional_img_vlm = numpy_to_base64(additional_rgb)
#     stage1_prompt_lst.append({
#         "type": "image_url",
#         "image_url": {"url": additional_img_vlm},
#     })
    
#     stage1_prompt_lst.append({"type": "text", "text": stage1_prompt})

#     stage1_headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json"
#     }

#     stage1_data = {
#         "model": "qwen25-vl",
#         "messages": [
#             {"role": "system", "content": "You are a helpful scene analysis expert."},
#             {
#                 "role": "user",
#                 "content": stage1_prompt_lst,
#             },
#         ],
#         "response_format": {"type": "json_object"},
#         "stream": False,
#         "temperature": 0.0,
#         "top_p": 0.95
#     }
    
#     try:
#         stage1_response = requests.post(url, headers=stage1_headers, data=json.dumps(stage1_data), verify=False)
        
#         if stage1_response.status_code != 200:
#             return False
            
#         stage1_content_json = stage1_response.json()
#         stage1_content = stage1_content_json["choices"][0]["message"]["content"]
        
#         try:
#             cleaned_content = stage1_content.strip()
#             if cleaned_content.startswith('```json'):
#                 cleaned_content = cleaned_content[7:]
#             if cleaned_content.endswith('```'):
#                 cleaned_content = cleaned_content[:-3]
#             cleaned_content = cleaned_content.strip()
            
#             stage1_results = json.loads(cleaned_content)
#         except json.JSONDecodeError as e:
#             return False

        
        
#         critical_desc_section = ""
#         if object_description:
#             critical_desc_section = f"""## CRITICAL OBJECT DESCRIPTION - {object_type.upper()}
# **VERY IMPORTANT**: The following description contains ESSENTIAL characteristics that MUST be considered when identifying a '{object_type}':

# **{object_description}**

# **This description is CRITICAL for accurate identification. Pay special attention to these specific features and requirements.**

# ---
# """
        
#         critical_check_section = f"**CRITICAL CHECK**: Before proceeding, ensure you have carefully reviewed the CRITICAL OBJECT DESCRIPTION above for '{object_type}'. These requirements are MANDATORY for accurate identification.\n\n" if object_description else ""
        
#         critical_verification_section = f"6. **CRITICAL FEATURE VERIFICATION**: For '{object_type}', verify that the object meets ALL the essential characteristics described in the CRITICAL OBJECT DESCRIPTION above. This is MANDATORY for accurate identification.\n7. " if object_description else "6. "
        
#         final_verification_section = f"**FINAL CRITICAL VERIFICATION**: Before making your final decision, ensure that your chosen object meets ALL the essential characteristics specified in the CRITICAL OBJECT DESCRIPTION for '{object_type}'. This is the final checkpoint for accurate identification.\n\n" if object_description else ""
        
#         requirements_check_section = f"4. **CRITICAL REQUIREMENTS CHECK**: Explicitly verify that your chosen object meets ALL the critical requirements for '{object_type}' as specified in the CRITICAL OBJECT DESCRIPTION above.\n5. " if object_description else "4. "
        
#         stage2_prompt = f"""
# You are performing final object confirmation based on scene analysis results and AI system belief information. Your primary mission is to identify the object(s) **highlighted by the green bounding box** by choosing the best label from a given list. Your final goal is to help determine if the object is a "{object_type}".

# ## Multi-View Image Analysis
# **IMPORTANT**: You have been provided with **{len(view_rgbs)} images** showing the same object from different perspectives. This multi-view approach is crucial for accurate object identification:

# 1. **View Quality Assessment**: Some images may have better viewing angles or clearer details than others. Pay special attention to images where the object is most clearly visible and identifiable.

# 2. **Cross-View Validation**: Compare the object's appearance across different images. Look for consistent features, shapes, and characteristics that appear in multiple views.

# 3. **Best View Selection**: If one or more images provide clearer evidence for object identification, prioritize those views in your analysis. Don't be misled by images with poor viewing angles or unclear details.

# 4. **Comprehensive Analysis**: Use the combination of all views to build a complete understanding of the object's structure, function, and category.

# {critical_desc_section}## Crucial Instructions
# 1. **Target Object Type is ONLY a Reference**: The "{object_type}" is just a proposed label to verify. Do NOT assume it's correct. You must make an independent assessment.
# 2. **Object vs. Non-Object First**: Before choosing from the list, you **MUST** first determine if the box highlights a distinct, three-dimensional object or just a static surface (like a floor, wall, ceiling) or an unrecognizable pattern.
# 3. **Independent Analysis**: Base your decision on what you actually see across all provided images, not on what the target type suggests.
# 4. **Belief-Guided Analysis**: Use the AI system's belief information to guide your analysis, but don't let it override your visual judgment.

# ## Context Information
# - **Proposed Label to Verify (for final output)**: "{object_type}"
# - **Candidate List (for your primary choice)**: [{similar_object_str}]
# - **Permitted Vocabulary**: [{vocabulary_str}]

# ## Scene Analysis Results (from previous VLM)
# - **Environment**: {stage1_results.get('environment', 'Unknown')}
# - **Scene Objects**: {stage1_results.get('scene_objects', [])}
# - **Green Box Object**: {stage1_results.get('green_box_object', 'Unknown')}
# - **Identified Object Type**: {stage1_results.get('object_type', 'Unknown')}
# - **Reasoning**: {stage1_results.get('reasoning', 'No reasoning provided')}

# ## Enhanced Reasoning Process: "Verify, Identify, and Conclude"

# ### Step 1: Verify if it's a Distinct Object
# 1. Look closely at the area in the green box. Is it a self-contained object with clear boundaries, depth, and form?
# 2. Or is it a flat surface, a part of a larger architectural feature, or just a texture?
# 3. If it is **NOT** a distinct object, your decision is made: choose the most appropriate label from the candidate list.

# ### Step 2: Candidate List Analysis
# 1. **Review the Candidate List**: [{similar_object_str}] - these are the most likely object types based on prior knowledge
# 2. **Consider Similar Objects**: The candidate list contains objects that are visually or functionally similar to the target type
# 3. **Use candidates as guidance**: These suggestions should inform your analysis, but don't let them override what you actually see

# ### Step 3: Multi-View Visual Analysis and Identification
# {critical_check_section}1. **Assess Image Quality**: Review all {len(view_rgbs)} images and identify which ones provide the clearest view of the object. Note any images with poor viewing angles, partial occlusion, or unclear details.

# 2. **Cross-View Feature Analysis**: Meticulously analyze the verifiable components of the object(s) across all images. Look for consistent features that appear in multiple views and ignore features that only appear in low-quality images.

# 3. **Best View Prioritization**: Focus your analysis on the images that provide the most complete and clear view of the object. If some images show the object more clearly than others, prioritize those views for your identification.

# 4. **Candidate List Reference**: Compare your visual observation with the **Candidate List**: [{similar_object_str}].
# {critical_verification_section}Choose the single best and most plausible label from THIS list to describe the object(s).
# {f"6. " if object_description else "5. "}State your choice clearly based on visual evidence from the best available views{f" and confirm that it meets the critical requirements for '{object_type}'" if object_description else ""}.

# ### Step 4: Final Decision and Justification
# {final_verification_section}1. **Synthesize Multi-View Evidence**: Make your final choice based on the strongest evidence from the best available views
# 2. **Quality-Based Reasoning**: Provide a concise, evidence-based reason for your choice, explicitly mentioning which images provided the clearest evidence
# 3. **Cross-View Validation**: Explain *why* you selected it based on its **verifiable parts** that appear consistently across multiple views
# {requirements_check_section}**Environmental Context**: Consider the environmental context from the scene analysis
# {f"6. " if object_description else "5. "}**View Quality Assessment**: If some images were unclear or provided poor viewing angles, explain how you handled this in your decision-making process

# ### Step 5: Output Format Decision
# 1. Now, consider your chosen best label from Step 4.
# 2. If your chosen best label **is** `"{object_type}"`, you will set `confirmed_object_type` to `"{object_type}"`.
# 3. If your chosen best label **is NOT** `"{object_type}"`, you will set `confirmed_object_type` to your chosen label.

# ## Output Format
# Respond with a strict JSON object containing:

# {{
#     "confirmed_object_type": "string",
#     "confidence_score": "float",
#     "reasoning": "string",
#     "multi_view_analysis": {{
#         "total_images": "integer",
#         "image_quality_assessment": "string",
#         "best_views_identified": "string",
#         "cross_view_consistency": "string",
#         "feature_analysis": "string"
#     }},
#     "decision_process": {{
#         "candidate_evaluation": "string",
#         "final_choice_rationale": "string",
#         "uncertainty_handling": "string"
#     }},
#     "critical_description_analysis": {{
#         "description_available": "boolean",
#         "key_requirements_met": "string",
#         "feature_verification": "string",
#         "compliance_assessment": "string"
#     }}
# }}

# ## Important Notes
# - **confirmed_object_type**: must be from the candidate list: [{similar_object_str}]
# - **confidence_score**: 0.0-1.0, your confidence level based on visual evidence
# - **reasoning**: comprehensive summary of your analysis process
# - **multi_view_analysis**: detailed analysis of the {len(view_rgbs)} images provided
#   - **total_images**: number of images analyzed
#   - **image_quality_assessment**: which images had the best/worst viewing angles
#   - **best_views_identified**: which specific images provided the clearest evidence
#   - **cross_view_consistency**: what features were consistent across multiple views
#   - **feature_analysis**: key visual features that led to your identification
# - **decision_process**: your step-by-step decision-making process
#   - **candidate_evaluation**: how you evaluated each candidate in the list
#   - **final_choice_rationale**: why you chose the specific object type
#   - **uncertainty_handling**: how you handled any uncertainties or ambiguities
# - **critical_description_analysis**: analysis of the critical object description (if available)
#   - **description_available**: whether a critical description was provided for the target object type
#   - **key_requirements_met**: which key requirements from the description were met or not met
#   - **feature_verification**: detailed verification of specific features mentioned in the description
#   - **compliance_assessment**: overall assessment of compliance with the critical description requirements

# Focus on making an independent, evidence-based decision based on visual analysis. Do NOT be biased by the proposed target type, but DO consider the candidate list as valuable guidance.
# """

#         stage2_data = {
#             "model": "qwen25-vl",
#             "messages": [
#                 {"role": "system", "content": "You are a helpful object confirmation expert."},
#                 {
#                     "role": "user",
#                     "content": [{"type": "text", "text": stage2_prompt}],
#                 },
#             ],
#             "response_format": {"type": "json_object"},
#             "stream": False,
#             "temperature": 0.0,
#             "top_p": 0.95
#         }
#         stage2_response = requests.post(url, headers=stage1_headers, data=json.dumps(stage2_data), verify=False)
#         if stage2_response.status_code != 200:
#             return False
            
#         stage2_content_json = stage2_response.json()
#         stage2_content = stage2_content_json["choices"][0]["message"]["content"]
        
#         try:
#             cleaned_content = stage2_content.strip()
#             if cleaned_content.startswith('```json'):
#                 cleaned_content = cleaned_content[7:]
#             if cleaned_content.endswith('```'):
#                 cleaned_content = cleaned_content[:-3]
#             cleaned_content = cleaned_content.strip()
            
#             stage2_results = json.loads(cleaned_content)
#         except json.JSONDecodeError as e:
#             return False
        
#         print(stage2_prompt)
#         print(f"[VLM Final] Qwen thinks this object is {stage2_results['confirmed_object_type']}")
#         print(f"[VLM Final] Confidence: {stage2_results['confidence_score']}")
#         print(f"[VLM Final] Reasoning: {stage2_results['reasoning']}")
        
#         if 'multi_view_analysis' in stage2_results:
#             mv_analysis = stage2_results['multi_view_analysis']
#             print(f"[VLM Final] Multi-view Analysis:")
#             print(f"  - Total images: {mv_analysis.get('total_images', 'N/A')}")
#             print(f"  - Best views: {mv_analysis.get('best_views_identified', 'N/A')}")
#             print(f"  - Cross-view consistency: {mv_analysis.get('cross_view_consistency', 'N/A')}")
        
#         if 'decision_process' in stage2_results:
#             decision_info = stage2_results['decision_process']
#             print(f"[VLM Final] Decision Process:")
#             print(f"  - Final choice rationale: {decision_info.get('final_choice_rationale', 'N/A')}")
        
#         if 'critical_description_analysis' in stage2_results:
#             desc_info = stage2_results['critical_description_analysis']
#             print(f"[VLM Final] Critical Description Analysis:")
#             print(f"  - Description available: {desc_info.get('description_available', 'N/A')}")
#             print(f"  - Key requirements met: {desc_info.get('key_requirements_met', 'N/A')}")
#             print(f"  - Feature verification: {desc_info.get('feature_verification', 'N/A')}")
#             print(f"  - Compliance assessment: {desc_info.get('compliance_assessment', 'N/A')}")

        
#         if stage2_results["confirmed_object_type"] == object_type:
#             print("[VLM Final] confirmed")
#             return True
#         else:
#             print("[VLM Final] not object type")
#             return False
            
#     except Exception as e:
#         return False



def vlm_belief_distribution(url, model, view_rgb, vocabulary, use_bbox=True, authorization=""):
    """Estimate a probability for each vocabulary class using the vision-language model."""
    
    vocabulary_str = ", ".join(f'"{item}"' for item in vocabulary)
    
    prompt = f"""
Your primary mission is to analyze the entire image and provide a belief distribution over the given vocabulary. You need to assign a probability score to each category based on what you observe in the image.

## Vocabulary
{vocabulary_str}

## Crucial Instructions
1. **Comprehensive Analysis**: Analyze the ENTIRE image, not just specific regions. Look for all possible objects and surfaces.
2. **Object vs. Non-Object Recognition**: Distinguish between distinct, three-dimensional objects and static surfaces (floors, walls, ceilings, textures).
3. **Multiple Object Handling**: If you see multiple instances of the same object type, consider them as a single category.
4. **Background Consideration**: Always include "background" as a possibility for areas that are not distinct objects.

## Critical Instructions for Probability Distribution
1. **Flat Distribution Requirement**: All probabilities should be relatively close to each other, similar to DINO detector outputs
2. **Maximum Difference Constraint**: The difference between the highest and lowest probability should NOT exceed 0.1
3. **No Zero Probabilities**: Every category must have a non-zero probability (minimum 0.01)
4. **Conservative Scoring**: Avoid extreme confidence - most probabilities should be in the range 0.05-0.15
5. **Subtle Differences**: Use small differences (0.01-0.05) to indicate slight preferences

## Mandatory Reasoning Process: "Scan, Analyze, Distribute, Justify"

### Step 1: Comprehensive Image Scan
1. **Scan the entire image systematically** - look at all regions, not just the center
2. **Identify all visible objects** - furniture, fixtures, surfaces, textures
3. **Note object characteristics** - size, shape, material, position, context
4. **Consider partial visibility** - objects that might be partially obscured

### Step 2: Object Analysis and Categorization
1. **For each visible object/area**, determine which vocabulary categories it could belong to
2. **Analyze verifiable components** - look for legs, surfaces, materials, shapes that match category definitions
3. **Consider context clues** - room type, object relationships, typical placements
4. **Handle ambiguous cases** - when objects could fit multiple categories, consider the most likely

### Step 3: Probability Distribution
1. **Assign base probabilities** to all categories (minimum 0.01 for each)
2. **Increase probabilities** for categories with strong visual evidence
3. **Maintain flat distribution** - keep differences small (0.01-0.05 between adjacent categories)
4. **Ensure normalization** - all probabilities must sum to 1.0

### Step 4: Justify Your Distribution
1. **Explain your reasoning** for the highest probability categories
2. **Acknowledge uncertainty** for lower probability categories
3. **Note any ambiguous cases** or partial visibility issues

## Probability Guidelines
- **Most likely category**: 0.12-0.15 (highest, but not dominant)
- **Secondary categories**: 0.08-0.12 (moderate confidence)
- **Less likely categories**: 0.05-0.08 (low but non-zero)
- **Background/uncertain**: 0.05-0.07 (always present)

## Output Format
You must respond with a JSON object containing:
- "belief_distribution": A dictionary mapping each vocabulary item to its probability
- "confidence": Overall confidence in the analysis (0.0 to 1.0)
- "reasoning": Detailed explanation of your analysis process

Example format (note the flat distribution):
{{
    "belief_distribution": {{
        "chair": 0.12,
        "table": 0.10,
        "sofa": 0.08,
        "background": 0.06,
        "monitor": 0.05
    }},
    "confidence": 0.75,
    "reasoning": "I can see furniture in the image with a chair being most prominent, a table partially visible, and some background surfaces. All categories have some possibility based on the visual evidence."
}}

Now analyze the image systematically, follow the "Scan, Analyze, Distribute, Justify" process, and provide your flat belief distribution.
"""
    
    img_vlm = numpy_to_base64(view_rgb)
    
    messages = [
        {"role": "system", "content": "You are a helpful visual analysis expert for object recognition."},
        {"role": "user", "content": prompt, "images": [img_vlm]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response_data = _post_ollama_chat(
            url,
            payload,
            authorization=authorization
        )
        content = response_data.get("message", {}).get("content", "")
        start = content.find('{')
        end = content.rfind('}')
        
        json_str = content[start:end+1]
        results = json.loads(json_str)
        
        belief_dist = results.get("belief_distribution", {})
        
        normalized_belief = {}
        for category in vocabulary:
            normalized_belief[category] = belief_dist.get(category, 0.0)
        
        total_prob = sum(normalized_belief.values())
        if total_prob > 0:
            for category in normalized_belief:
                normalized_belief[category] /= total_prob
        else:
            uniform_prob = 1.0 / len(vocabulary)
            for category in vocabulary:
                normalized_belief[category] = uniform_prob

        belief_tensor = torch.tensor([normalized_belief[category] for category in vocabulary], dtype=torch.float32)
        
        return belief_tensor
        
    except Exception as e:
        uniform_prob = 1.0 / len(vocabulary)
        fallback_tensor = torch.full((len(vocabulary),), uniform_prob, dtype=torch.float32)
        return fallback_tensor



def active_perception_confirmation(url, model, view_rgbs, object_type, similar_object, vocabulary, authorization=""):
    """Confirm an object using multiple observations."""
    print(f"[Active Perception VLM] Checking {len(view_rgbs)} images for object type: {object_type}")
    print(f"[Active Perception VLM] Similar objects to compare: {similar_object}")
    
    if len(view_rgbs) == 0:
        print("[Active Perception VLM] No images provided, returning False")
        return False
    
    if object_type not in similar_object:
        similar_object.append(object_type)
    if object_type == "table" and "sink" in similar_object:
        del similar_object[similar_object.index("sink")]
    random.shuffle(similar_object)
    similar_object_str = ", ".join(f'"{s}"' for s in similar_object)
    vocabulary_str = ", ".join(f'"{v}"' for v in vocabulary)

    prompt = f"""
You are analyzing a set of images captured through **Active Perception** - a process where an agent deliberately moves its camera to get better views of an object that was initially only partially visible (its bounding box was near the image boundary).

## Your Mission: Multi-View Object Confirmation

Your goal is to determine if the object visible across these multiple viewpoints is a "{object_type}".

## Critical Context: Active Perception Scenario

- **These images were captured specifically because the initial view was incomplete** (the object's bounding box was cut off by image boundaries)
- **Each image shows the same object from different angles** (left, right, up, down turns)
- **Your task is to synthesize information from ALL viewpoints** to make a confident identification
- **The green bounding box in each image highlights the same target object** from different perspectives

## Key Instructions

1. **Multi-View Synthesis**: Analyze ALL images together. Look for consistent features across different angles.
2. **Boundary Context**: Remember that the initial view was incomplete - use the additional viewpoints to fill in missing information.
3. **Consistency Check**: The same object should be identifiable across all angles, even if partially visible in individual images.
4. **Holistic Assessment**: Make your final decision based on the complete picture formed by all viewpoints.

## Contextual Information
- **Target Object Type**: "{object_type}"
- **Candidate List**: [{similar_object_str}]
- **Permitted Vocabulary**: [{vocabulary_str}]

## Analysis Process

### Step 1: Multi-View Feature Analysis
For each image, briefly note:
- What parts of the object are visible
- Key identifying features you can observe
- Any unique characteristics that help with identification

### Step 2: Cross-View Consistency
- Do the features observed across different angles belong to the same object?
- Are there consistent identifying characteristics across all viewpoints?
- Does the object maintain its fundamental shape and function across angles?

### Step 3: Comprehensive Identification
Based on your multi-view analysis:
- What is the most likely object type?
- How confident are you in this identification?
- Does this match the target object type "{object_type}"?

## Output Requirement
- You must respond ONLY with a single, valid JSON object. Do not include any text or markdown formatting.
- It is STRICTLY FORBIDDEN to wrap the JSON in markdown blocks like ```json ... ```.

Your response must strictly follow this format:
{{
  "confirmed_object_type": "string",
  "confidence_score": "float",
  "reasoning": "string"
}}

## Important Notes
- **confirmed_object_type**: the specific object type you identified (must be from the candidate list)
- **confidence_score**: your confidence level (0.0 = very uncertain, 1.0 = completely certain)
- **reasoning**: explain how the multiple viewpoints helped you make this decision

Focus on the comprehensive view provided by the active perception process, not just individual images.
"""

    image_payloads = [numpy_to_base64(view_rgb) for view_rgb in view_rgbs]
    messages = [
        {"role": "system", "content": "You are a helpful object identification expert for active perception."},
        {"role": "user", "content": prompt, "images": image_payloads}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        content_json = _post_ollama_chat(
            url,
            payload,
            authorization=authorization
        )
        content = content_json.get("message", {}).get("content", "")
        results = json.loads(content)
        
        print(f"[Active Perception VLM] Qwen thinks this object is {results['confirmed_object_type']} in {similar_object} and the reason is {results['reasoning']}")
        
        if results["confirmed_object_type"] == object_type:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"[Active Perception VLM] VLM query error: {e}")
        return False


if __name__ == '__main__':
    # client = OpenAI(
    #     api_key=openai_api_key,
    #     base_url=openai_api_base,
    # )
    openai_api_base = os.environ.get("E2NAV_CHAT_URL", "http://localhost:11434/api/chat")
    model = "Qwen3-VL-235B-A22B-Instruct"
    authorization = os.environ.get("E2NAV_AUTHORIZATION", "")

    img = "temp/2.png"
    pil_image = Image.open(img)
    rgb = np.array(pil_image)
    with open('utils/classes/classes.txt', 'r') as file:
        vocabulary = [line.rstrip('\n') for line in file]
    
    # class object_node():
    #     def __init__(self, class_name) -> None:
    #         self.class_name = class_name
    
    # obj1 = object_node("sink")
    # obj2 = object_node("mirror")
    # obj3 = object_node("window")
    # obj4 = object_node("curtain")
    
    # img_path_1 = "temp/2.png"

    # pil_image_1 = Image.open(img_path_1)
    # rgb_1 = np.array(pil_image_1)

    # optional_room_list = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "home gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
    

    # room, obj_node = room_type(openai_api_base, rgb_1, optional_room_list, [obj1, obj2, obj3, obj4])
    # print(room, obj_node.class_name if obj_node is not None else obj_node)


    # optional_room_dict = {'living room': 1, 'dining room': 1, 'bedroom': 1, 'kitchen': 1, 'bathroom': 5, 'entryway': 1, 'study': 1, 'home gym': 1, 'storage room': 1, 'basement': 1, 'balcony': 1, 'garden': 1, 'walk-in closet': 1}
    # found, possbility = object_detection(openai_api_base, rgb, "bathtub", optional_room_dict, [obj1, obj2, obj3, obj4])
    # print(f"check this:{found}, {possbility}")
    
    
    # confirming_img = "temp/bbox/scene_bbox_0004.png"
    

    # pil_image = Image.open(confirming_img)
    # rgb = np.array(pil_image)
    
    # answer = object_comparison(client, rgb, ["cabinet", "chest_of_drawers"])
    # print("debug1:", answer)
    authorization = os.environ.get("E2NAV_AUTHORIZATION", "")
    answer = object_confirmed(
                        url=openai_api_base,
                        model=model,
                        view_rgb=rgb,
                        object_type='television',
                        similar_object=["picture", "mirror", "window"],
                        vocabulary=vocabulary,

                        authorization=authorization
                    )
    
    # answer = object_confirmed(openai_api_base, model, rgb, 'television', ["picture", "mirror", "window"], vocabulary, "Commonly positioned in living rooms or bedrooms, a television features a large rectangular screen framed by a slim border, often mounted on a wall or stand.", authorization)
    print("debug1:", answer)
    # answer = object_confirmed(openai_api_base, rgb, 'counter', ["table", "cabinet"])
    # print("debug2:", answer)
    # answer = stairs_class(client, rgb)
    # print(answer)
    
    
    # rgb_sequence = []

    # pil_image = Image.open(confirming_img)
    # rgb_sequence.append(np.array(pil_image))

    # pil_image = Image.open(confirming_img)
    # rgb_sequence.append(np.array(pil_image))

    # pil_image = Image.open(confirming_img)
    # rgb_sequence.append(np.array(pil_image))
    
    # results = object_final_confirmed(openai_api_base, rgb_sequence, "monitor", ['book', 'counter', 'window', 'table', 'monitor', 'chest_of_drawers', 'picture', 'mirror'], vocabulary)
    # print(results)
    
    
    # test_image = "temp/2.png"
    # pil_image = Image.open(test_image)
    # rgb = np.array(pil_image)
    # result = nearby_confirmed(openai_api_base, rgb, "toilet")
    # print(result)
    
    # with open("utils/classes/classes.txt", 'r') as file:
    #     vocabulary = [line.rstrip('\n') for line in file]
        
    
    # try:
    #     pil_image = Image.open(image_path)
    #     test_rgb = np.array(pil_image)
        
    #     result = vlm_belief_distribution(openai_api_base, test_rgb, vocabulary)
        
    #     print("VLM Belief Distribution Result:")
    #     print(vocabulary)
    #     print(result)
        
    # except Exception as e:
    #     print(f"Error in test: {e}")


    