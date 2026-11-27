import json
import os
import re

import requests

from utils.chat_client import post_chat as _post_ollama_text


def _extract_llm_content(response_data: dict) -> str:
    """
    Helper to extract textual content from either Ollama-style responses
    (top-level `message`) or OpenAI-style responses (`choices[0].message`).
    """
    if not isinstance(response_data, dict):
        return ""
    
    message = response_data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    
    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            choice_msg = first_choice.get("message", {})
            if isinstance(choice_msg, dict):
                content = choice_msg.get("content")
                if isinstance(content, str):
                    return content
            # Some providers may return `content` directly inside each choice
            choice_content = first_choice.get("content")
            if isinstance(choice_content, str):
                return choice_content
    
    # Fallback: some APIs might return a top-level "content"
    content = response_data.get("content")
    if isinstance(content, str):
        return content
    
    return ""


def room_rank(url, model, optional_room_list, obj_goal, authorization=""):
    # client = OpenAI(
    #     api_key=openai_api_key,
    #     base_url=openai_api_base,
    # )
   
    prompt = f'''You are an automated JSON generation system. Your sole task is to evaluate the likelihood of finding an object in a list of rooms and return the result in a strict JSON format.

                Analyze the following inputs:
                - Object to find (obj_goal): "{obj_goal}"
                - List of rooms (optional_room_list): {optional_room_list}

                Based on the inputs, perform the following internal reasoning (do NOT include this reasoning in your output):
                1.  Consider the typical use and properties of the object: "{obj_goal}".
                2.  For each room in the list, evaluate its primary function.
                3.  Assign a likelihood score from 1 (extremely unlikely) to 5 (extremely likely) for finding the object in each room.

                Your output MUST be a single, valid JSON object and nothing else.
                - The response MUST start with `{{` and end with `}}`.
                - Do NOT include any text, explanations, apologies, or markdown formatting like ```json ... ``` before or after the JSON object.

                The JSON object MUST conform to this exact structure:
                - It must be a dictionary where keys are the room names from the input list and values are their corresponding scores.
                - Each key MUST be a string that exactly matches a room name from the `optional_room_list`.
                - Each value MUST be an integer between 1 and 5.

                Example:
                If obj_goal is "keys" and optional_room_list is ["kitchen", "bedroom", "hallway"], the correct and only valid output is:
                {{
                    "kitchen": 3,
                    "bedroom": 2,
                    "hallway": 5
                }}

                CRITICAL: Generate ONLY the JSON object.
            '''

    messages = [
        {
            "role": "user",
            "content": prompt
        }
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
        response_data = _post_ollama_text(
            url,
            payload,
            authorization=authorization
        )
        content = _extract_llm_content(response_data)

        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
            try:
                results = json.loads(json_str)
                return results
            except json.JSONDecodeError as e:
                print(f"error2: {e}")
                return {}
        else:
            return {}

    except requests.exceptions.RequestException as e:
        print(f"error3: {e}") 
        return {} # Return a safe default for network issues


def llm_auxiliary_frontier(client_url, model, obj_goal, landmark_scenes, authorization=""):

    # The prompt is constructed as an f-string.
    # It clearly defines the role, task, input format, reasoning process, and output requirements.
    prompt = f"""
You are a robot navigation assistant. Score each landmark conservatively using both the room context and the observed objects.

## Scoring criteria

1. Treat `room_name`, when available, as a strong prior (for example, kitchen or living room).
2. Use `observed_objects` to assess whether the objects fit the target or room.
3. Equally plausible landmarks may share a middle score around 5.
4. Use 7-10 when both sources strongly support the target, 4-6 for partial evidence,
   and 0-3 for weak or contradictory evidence.
5. Reserve 9-10 for near certainty and 0-1 for near impossibility.

---

## Your Task and Inputs

-   **Task**: Apply the Golden Rule to the landmarks below. Decide if any single landmark meets the extreme criteria for a non-uniform score.
-   **Target Object**: '{obj_goal}'
-   **Landmark Observations** (each entry contains a `room_name` field and an `observed_objects` list with `object_name`, `distance`, and `direction`):
    ```json
    {json.dumps(landmark_scenes, indent=2)}
    ```

## Your Reasoning Process (Strictly follow this)

1. For each landmark, infer the most likely room from `room_name` (if available) and `observed_objects`.
2. Score how well the room fits `'{obj_goal}'` using the ranges above.
3. Use equal middle scores for equally plausible landmarks. Widen the gap when evidence is particularly strong.

---

## Examples of Applying the Golden Rule

### EXAMPLE 1: UNIFORM SCORES (The Common Case)
-   **Target**: `Laptop`
-   **Landmark "0"**: `Sofa`, `CoffeeTable` (Suggests a Living Room)
-   **Landmark "1"**: `Bed`, `Pillow` (Suggests a Bedroom)
-   **Reasoning**: A laptop is very plausible in both a living room and a bedroom. Neither location is an "indisputable" or "irrefutable" guarantee. Therefore, the Golden Rule applies.
-   **Correct Output**: `{{"0": 5, "1": 5}}`

### EXAMPLE 2: DIFFERENTIATED SCORES (The Rare Exception)
-   **Target**: `Soap`
-   **Landmark "0"**: `Sink`, `Towel`, `Toilet` (Indisputably a Bathroom)
-   **Landmark "1"**: `Bookshelf`, `DeskLamp` (Indisputably a Study/Office)
-   **Reasoning**: The evidence for `Soap` in a bathroom area (Landmark "0") is irrefutable. The evidence against it in a study is also overwhelming. The extreme certainty justifies breaking the uniformity rule.
-   **Correct Output**: `{{"0": 10, "1": 0}}`

---

## Final Output Requirements

-   Your final response **MUST** be a single, raw JSON object and nothing else.
-   Start with `{{` and end with `}}`. No markdown, no explanations.
-   Include a `scores` dictionary and a concise `reason` string explaining the room and object evidence.
    {{
        "scores": {{"0": 5, "1": 5}},
        "reason": "Explain in <40 words why the scores were chosen."
    }}
-   `scores` keys are landmark ID strings, values are numbers from 0 to 10.
-   `reason` summarizes how room context and observed objects influenced your decision.

Final JSON Output:
"""

    messages = [
        {"role": "system", "content": "You are an expert robot navigator and scene analyst."},
        {"role": "user", "content": prompt}
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
        response = _post_ollama_text(
            client_url,
            payload,
            authorization=authorization,
            timeout=60
        )
        content = _extract_llm_content(response)
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("LLM response did not contain a valid JSON object.")
            
        json_string = json_match.group(0)
        parsed_output = json.loads(json_string)
        
        reason = ""
        scores_payload = parsed_output
        if isinstance(parsed_output, dict):
            reason = parsed_output.get("reason", "") or ""
            if "scores" in parsed_output:
                scores_payload = parsed_output.get("scores", {})
        
        scores = {int(k): float(v) for k, v in scores_payload.items()}
        return scores, reason

    except requests.Timeout as timeout_err:
        print(f"[llm_auxiliary_frontier] LLM request timed out: {timeout_err}. Returning default scores.")
    except (requests.RequestException, json.JSONDecodeError, AttributeError, ValueError, KeyError) as e:
        print(f"Error during LLM call or parsing: {e}. Returning default scores.")

    default_scores = {int(key): 1.0 for key in landmark_scenes.keys()}
    return default_scores, ""

# def llm_auxiliary_frontier(url, obj_goal, landmark_object_matching):
#     # client = OpenAI(
#     #     api_key=openai_api_key,
#     #     base_url=openai_api_base,
#     # )
    
#     prompt = f"""
# You are an expert indoor scene understanding assistant for a robot. Your task is to evaluate several potential exploration landmarks and assign a "value score" to each. This score represents the likelihood that exploring in the direction of that landmark will lead the robot to find a specific `target_object`.
# **Input:**
# 1.  **`target_object`**: The type of object the robot is searching for.
#     `target_object`: '{obj_goal}'

# 2.  **`landmark_observations`**: A JSON dictionary where keys are landmark indices and values are lists of observed objects.
#     `landmark_observations`:
#     ```json
#     {landmark_object_matching}
#     ```

# **Your Goal:**

# For each landmark index, assign a score from 0 to 10.
# - A score of **10** means it's extremely likely the `target_object` will be found there.
# - A score of **5** means a moderate or plausible chance.
# - A score of **0** means it's highly unlikely.

# **Reasoning Process (Think step-by-step for each landmark):**

# 1.  **Identify the `target_object`**: What are we looking for? (e.g., '{obj_goal}')
# 2.  **For each landmark:**
#     a.  **Examine Observed Objects**: Look at the list of objects for this landmark.
#     b.  **Infer Scene Context**: What kind of room is this? (e.g., kitchen, bedroom).
#     c.  **Assess Likelihood of Target**: Based on the context, how likely is the `target_object` ('{obj_goal}') to be found here?

# **Output Format:**

# Your final output **MUST** be a single, valid JSON object.
# - The keys of the JSON object **MUST be STRINGS** representing the integer landmark indices (e.g., "0", "1", "2").
# - The values **MUST be numbers** (integer or float) between 0 and 10.
# - Do **NOT** include any extra text, explanations, or markdown fences (like ```json). The response must start with `{{` and end with `}}`.

# **Example of a valid final output:**
# ```json
# {{"0": 9, "1": 2, "2": 5}}
# ```

# **Final Output (A single valid JSON object only):**
#             """
        
#     headers = {
#     "Accept": "application/json",
#     "Content-Type": "application/json"
#     }
    
#     data = {
#         "model": "qwen25-vl",
#         "messages": [
#             {"role": "system", "content": "You are a helpful room classification expert."},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                 ],
#             },
#         ],
#         "response_format": {"type": "json_object"},
#         "stream": False,
#         "temperature": 0.0,
#         "top_p": 0.95
#     }
#     response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    
#     try:
#         content = response.json()["choices"][0]["message"]["content"]
#         json_match = re.search(r'\{.*\}', content, re.DOTALL)
#         json_string = json_match.group(0)
#         parsed_scores = json.loads(json_string)
#         scores = {int(k): float(v) for k, v in parsed_scores.items()}

#     except (json.JSONDecodeError, AttributeError, ValueError) as e:
#         scores = landmark_object_matching
#         for key in scores.keys():
#             scores[key] = 1
            
#     return scores

if __name__ == '__main__':
    openai_api_base = os.environ.get("E2NAV_CHAT_URL", "http://localhost:11434/api/chat")
    model = "Qwen3-VL-235B-A22B-Instruct"
    authorization = os.environ.get("E2NAV_AUTHORIZATION", "")
    goal_object = "cabinet"
    optional_room_list = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
    result = room_rank(openai_api_base, model, optional_room_list, goal_object, authorization=authorization)
    print(result)
    
    TARGET_OBJECT = "laptop"

    SAMPLE_LANDMARK_SCENES = {
        0: {
            "room_name": "living room",
            "observed_objects": [
                {"object_name": "Sofa", "distance": "2.0m", "direction": "front-left"},
                {"object_name": "CoffeeTable", "distance": "1.2m", "direction": "front"},
                {"object_name": "Television", "distance": "3.0m", "direction": "front"},
                {"object_name": "Bookshelf", "distance": "2.8m", "direction": "left"},
                {"object_name": "FloorLamp", "distance": "1.9m", "direction": "back-right"},
            ]
        },
        1: {
            "room_name": "living room",
            "observed_objects": [
                {"object_name": "Armchair", "distance": "1.4m", "direction": "right"},
                {"object_name": "SideTable", "distance": "1.0m", "direction": "front-right"},
                {"object_name": "TVStand", "distance": "2.6m", "direction": "front"},
                {"object_name": "PottedPlant", "distance": "1.7m", "direction": "left"},
                {"object_name": "Rug", "distance": "0.5m", "direction": "center"},
            ]
        },
        2: {
            "room_name": "living room",
            "observed_objects": [
                {"object_name": "SectionalSofa", "distance": "2.3m", "direction": "front-left"},
                {"object_name": "CoffeeTable", "distance": "1.1m", "direction": "front"},
                {"object_name": "WallArt", "distance": "3.5m", "direction": "front"},
                {"object_name": "Speaker", "distance": "2.0m", "direction": "left"},
                {"object_name": "Curtains", "distance": "3.0m", "direction": "back"},
            ]
        },
        3: {
            "room_name": "bedroom",
            "observed_objects": [
                {"object_name": "Bed", "distance": "1.0m", "direction": "front"},
                {"object_name": "Nightstand", "distance": "1.3m", "direction": "front-right"},
                {"object_name": "Lamp", "distance": "1.1m", "direction": "right"},
                {"object_name": "Wardrobe", "distance": "2.4m", "direction": "left"},
                {"object_name": "Dresser", "distance": "2.0m", "direction": "front-left"},
            ]
        },
        4: {
            "room_name": "kitchen",
            "observed_objects": [
                {"object_name": "Sink", "distance": "1.5m", "direction": "left"},
                {"object_name": "Refrigerator", "distance": "2.2m", "direction": "front"},
                {"object_name": "Microwave", "distance": "1.8m", "direction": "front-right"},
                {"object_name": "Stove", "distance": "1.4m", "direction": "front"},
                {"object_name": "KitchenIsland", "distance": "0.9m", "direction": "center"},
            ]
        },
    }

    print("="*50)
    print("        LLM Landmark Scoring Test Script")
    print("="*50)

    print("\n[INPUT] Sending the following data to the LLM:")
    print(f"  - Target Object: '{TARGET_OBJECT}'")
    print(f"  - Landmark Scenes:")
    print(json.dumps(SAMPLE_LANDMARK_SCENES, indent=4, ensure_ascii=False))

    print("\n[INFO] Calling the LLM... please wait.")
    llm_scores, llm_reason = llm_auxiliary_frontier(
        openai_api_base,
        model,
        TARGET_OBJECT,
        SAMPLE_LANDMARK_SCENES,
        authorization=authorization
    )

    print("\n[OUTPUT] Received scores from LLM:")
    print("-" * 30)
    if llm_reason:
        print(f"[LLM Reason] {llm_reason}")

    if llm_scores:
        for landmark_id, score in sorted(llm_scores.items()):
            print(f"Landmark ID: {landmark_id}")
            print(f"  -> LLM Score: {score:.1f} / 10.0")
            scene_desc = SAMPLE_LANDMARK_SCENES.get(landmark_id, {})
            objects = scene_desc.get("observed_objects", [])
            print(f"  -> Observed Scene: {[obj['object_name'] for obj in objects]}")
            print("-" * 30)
    else:
        print("No scores were returned.")

    print("\nTest finished.")
