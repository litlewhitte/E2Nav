from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download  # 需要先安装 huggingface_hub
import re
import ast
import requests
import json
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://qwen2-5-72b-128k.api.sii.edu.cn/v1/chat/completions"

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

    headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
    }
    if authorization:
        headers["Authorization"] = authorization
    
    data = {
        "model": model,
        "messages": [
            # {"role": "system", "content": "You are a helpful room classification expert."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "temperature": 0.0,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        # 1. Check for HTTP errors
        if response.status_code != 200:
            print("response.status_code != 200")
            return {}  # Return a safe default value

        # 2. Try to parse the top-level API response
        try:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"error1: {e}")
            return {} # Return a safe default value

        # 3. Clean and parse the model's content string
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
            try:
                results = json.loads(json_str)
                # Optional: Add a check here to validate the structure of the returned dict
                # e.g., ensure all keys are in optional_room_list and values are ints 1-5.
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
You are a robot navigation assistant with a strict, conservative scoring protocol. Your absolute top priority is to avoid false positives and weak guesses.

## THE GOLDEN RULE: UNIFORMITY IS YOUR DEFAULT (NON-NEGOTIABLE)

**This is the most important instruction. Read it carefully.**

1.  Your primary, default, and most common action is to assign the **exact same score (5)** to **ALL** landmarks. This indicates that no single location is a guaranteed winner.
2.  You are **ONLY** permitted to break this uniformity rule and give a different, high score (9-10) to a landmark if the evidence for the `target_object`'s presence is **INDISPUTABLE and IRREFUTABLE**.
3.  **What is "Indisputable Evidence"?** It means a near-100% certainty. The combination of objects points to one, and only one, logical conclusion. A "strong hint," "high probability," or a "very good guess" is **NOT ENOUGH** to break the uniformity rule. If you have any doubt, however small, you **MUST** return to Rule #1 and assign 5 to all.
4.  Giving different scores on anything less than irrefutable evidence is considered a **FAILURE** of your task.

---

## Your Task and Inputs

-   **Task**: Apply the Golden Rule to the landmarks below. Decide if any single landmark meets the extreme criteria for a non-uniform score.
-   **Target Object**: '{obj_goal}'
-   **Landmark Observations**:
    ```json
    {json.dumps(landmark_scenes, indent=2)}
    ```

## Your Reasoning Process (Strictly follow this)

1.  Scan all landmarks.
2.  Ask yourself: "Is there one, and ONLY one, landmark where the presence of '{obj_goal}' is virtually guaranteed, with no other plausible alternative?"
    -   **If the answer is YES**: Assign that landmark a 9 or 10. Assign all other landmarks a very low score (0-2).
    -   **If the answer is NO** (This will be the most common outcome): Assign the baseline score of **5** to **ALL** landmarks without exception.

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
-   Keys are landmark ID strings, values are numbers from 0 to 10.

Final JSON Output:
"""

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    if authorization:
        headers["Authorization"] = authorization

    data = {
        "model": model,  # Or your preferred model
        "messages": [
            {"role": "system", "content": "You are an expert robot navigator and scene analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "temperature": 0.0,
    }

    try:
        # Note: In production, `verify=False` should be used with caution.
        response = requests.post(client_url, headers=headers, data=json.dumps(data), verify=False, timeout=20)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        content = response.json()["choices"][0]["message"]["content"]
        
        # The json_object mode should make this robust, but a regex fallback is safe.
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("LLM response did not contain a valid JSON object.")
            
        json_string = json_match.group(0)
        parsed_scores = json.loads(json_string)
        
        # Convert keys to int and values to float for consistency
        scores = {int(k): float(v) for k, v in parsed_scores.items()}
        return scores

    except (requests.RequestException, json.JSONDecodeError, AttributeError, ValueError, KeyError) as e:
        print(f"Error during LLM call or parsing: {e}. Returning default scores.")
        # Fallback strategy: if LLM fails, return a default score for each landmark.
        # A score of 1 is neutral and indicates "low but non-zero possibility".
        default_scores = {int(key): 1.0 for key in landmark_scenes.keys()}
        return default_scores

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
    goal_object = "cabinet"
    optional_room_list = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
    result = room_rank(openai_api_base, optional_room_list, goal_object)
    print(result)
    
    # TARGET_OBJECT = "sink"

    # # We have three landmarks with different surrounding objects
    # SAMPLE_LANDMARK_SCENES = {
    #     # Landmark 0: Clearly a kitchen area. High chance of finding a mug.
    #     0: [
    #         {"object_name": "CounterTop", "distance": "1.5m", "direction": "in_front"},
    #         {"object_name": "CoffeeMachine", "distance": "1.8m", "direction": "to_the_left"},
    #         {"object_name": "Sink", "distance": "2.5m", "direction": "in_front"}
    #     ],
    #     # Landmark 1: An office/study area. Plausible to find a mug.
    #     1: [
    #         {"object_name": "Desk", "distance": "1.0m", "direction": "in_front"},
    #         {"object_name": "Chair", "distance": "1.2m", "direction": "behind"},
    #         {"object_name": "Book", "distance": "1.1m", "direction": "on_the_right"}
    #     ],
    #     # Landmark 2: A bathroom. Very low chance of finding a mug.
    #     2: [
    #         {"object_name": "Toilet", "distance": "2.0m", "direction": "to_the_right"},
    #         {"object_name": "Shower", "distance": "3.0m", "direction": "in_front"},
    #         {"object_name": "Towel", "distance": "1.5m", "direction": "on_the_left"}
    #     ]
    # }

    # print("="*50)
    # print("        LLM Landmark Scoring Test Script")
    # print("="*50)

    # print("\n[INPUT] Sending the following data to the LLM:")
    # print(f"  - Target Object: '{TARGET_OBJECT}'")
    # print(f"  - Landmark Scenes:")
    # print(json.dumps(SAMPLE_LANDMARK_SCENES, indent=4))

    # # --- 4. CALL THE FUNCTION AND GET RESULTS ---
    # print("\n[INFO] Calling the LLM... please wait.")
    # llm_scores = llm_auxiliary_frontier(openai_api_base, TARGET_OBJECT, SAMPLE_LANDMARK_SCENES)
    # print("zqh", llm_scores)
    # print("***")

    # # --- 5. DISPLAY THE RESULTS ---
    # print("\n[OUTPUT] Received scores from LLM:")
    # print("-" * 30)
    # if llm_scores:
    #     for landmark_id, score in sorted(llm_scores.items()):
    #         print(f"Landmark ID: {landmark_id}")
    #         print(f"  -> LLM Score: {score:.1f} / 10.0")
    #         print(f"  -> Observed Scene: {[obj['object_name'] for obj in SAMPLE_LANDMARK_SCENES[landmark_id]]}")
    #         print("-" * 30)
    # else:
    #     print("No scores were returned.")

    # print("\nTest finished.")