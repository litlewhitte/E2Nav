# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import json

def object_filter(vlm_model, vlm_processor, detections):
    obj_lst = []
    for idx, detection in enumerate(detections):
        label_str = detection['label']
        try:
            class_name = label_str.split('(')[0].strip()
            score = float(label_str.split('(')[1].rstrip(')'))
        except:
            class_name = "unknown"
            score = 0.0
            
        obj_lst.append(class_name)
    
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", 
             "text": 
                f'''
                You are a JSON output assistant. Follow these rules:
                1. Analyze this list: {obj_lst}
                2. Identify elements that are **Non-items** (scenes/rooms/abstract concepts).
                3. Make sure that the items you have not identified is furniture or tiny objects
                4. Return JSON **strictly** formatted as:
                {{"Non-items": [id1, id2,...]}} 
                - Use 0-based indexes between 0-{len(obj_lst)-1}
                - No explanations, only valid JSON
                - Keys must be double-quoted

                Example for ["Beach", "Laptop", "Conference Room", "Potted Plant"]:
                {{"Non-items": [0, 2]}}'''},
            ],
    }]


    # Preparation for inference
    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = vlm_model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        num_beams=1,
        eos_token_id=vlm_processor.tokenizer.eos_token_id,
        pad_token_id=vlm_processor.tokenizer.pad_token_id,
        temperature=0.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    vlm_response = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    idx_lst = parse_vlm_response(vlm_response)
    for index in sorted(idx_lst, reverse=True):
        del detections[index]
    print(obj_lst)
    print(idx_lst)
        
    return detections

def parse_vlm_response(response: list) -> list:
    raw_str = response[0]
    
    json_str = re.sub(r'[\s\n]+', ' ', raw_str)
    json_match = re.search(r'\{.*\}', json_str)
    if not json_match:
        return []
    
    try:
        repaired = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_match.group())
        data = json.loads(repaired)
        
        non_items = data.get("Non-items", data.get("non_items", []))
        
        valid_indices = [int(idx) for idx in non_items if isinstance(idx, (int, float)) and idx >= 0]
        return valid_indices
    except Exception as e:
        print(f"Parsing failed after repair attempt: {e}")
        return []
    
def close_vocabulary_filter(detections, vocabulary):
    for idx in reversed(range(len(detections))):
        detection = detections[idx]
        label_str = detection['label']
        try:
            class_name = label_str.split('(')[0].strip()
            score = float(label_str.split('(')[1].rstrip(')'))
        except:
            class_name = "unknown"
            score = 0.0
        if class_name not in vocabulary or class_name in ["wall", "floor", "ceiling", "pillar"]:
            del detections[idx]
    return detections