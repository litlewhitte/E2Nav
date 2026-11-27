import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# --- Configuration ---
MODEL_PATH = "Qwen/Qwen2.5-VL-72B-Instruct"
CACHE_DIR = "models/qwenVL_72B"
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16
ATTN_IMPLEMENTATION = None # Set to "flash_attention_2" if installed and desired

# --- Global Variables ---
app = FastAPI()
model: Optional[Qwen2_5_VLForConditionalGeneration] = None
processor: Optional[AutoProcessor] = None

# --- Pydantic Models for API Request/Response ---
class MessageContentItem(BaseModel):
    type: str
    image: Optional[str] = None # Can be URL, Base64 Data URI, or local path (server needs to handle)
    text: Optional[str] = None

class Message(BaseModel):
    role: str
    content: List[MessageContentItem]

class VLMTInferRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 128
    temperature: Optional[float] = None
    do_sample: Optional[bool] = None

class VLMTInferResponse(BaseModel):
    generated_text: str

# --- FastAPI Event Handling: Load Model and Processor on Startup ---
@app.on_event("startup")
async def load_resources():
    global model, processor
    print(f"Loading processor: {MODEL_PATH}")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        print("Loaded processor.")
    except Exception as e:
        print(f"Processor loading failed: {e}")
        raise RuntimeError(f"Cannot load processor: {e}")

    print(f"Loading model: {MODEL_PATH}")
    try:
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": TORCH_DTYPE,
            "device_map": DEVICE_MAP,
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
        }
        if ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
            print(f"Attention implementation: {ATTN_IMPLEMENTATION}")

        print(f"Model loading arguments: {model_kwargs}")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            **model_kwargs
        )
        model.eval()
        print("Loaded model.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Cannot load model: {e}")

# --- API Endpoint for Inference ---
@app.post("/infer", response_model=VLMTInferResponse)
async def infer(request: VLMTInferRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model or processor unavailable. Check startup logs and retry.")

    print(f"Inference request: max_new_tokens='{request.max_new_tokens}'")

    try:
        messages_as_dicts = []
        for msg_pydantic in request.messages:
            processed_content = []
            if msg_pydantic.content:
                for content_item_pydantic in msg_pydantic.content:
                    processed_content.append(content_item_pydantic.model_dump(exclude_none=True))
            
            messages_as_dicts.append({
                "role": msg_pydantic.role,
                "content": processed_content
            })
        
        # print(f"DEBUG: messages_as_dicts for qwen_vl_utils: {json.dumps(messages_as_dicts, indent=2)}")

        text_prompt_for_template = processor.apply_chat_template(
            messages_as_dicts,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_as_dicts)
        
        inputs = processor( 
            text=[text_prompt_for_template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available() and DEVICE_MAP == "auto":
             inputs = inputs.to(model.device)
        elif torch.cuda.is_available():
             inputs = inputs.to("cuda")
        else:
             inputs = inputs.to("cpu")


        print("Generating text...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
            )

        input_ids_on_device = inputs.input_ids.to(generated_ids.device)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_on_device, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False # Qwen's recommendation
        )[0]

        print(f"Generated text: {output_text}")
        return VLMTInferResponse(generated_text=output_text)

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000) # Default port, can be changed
    args = parser.parse_args()
    
    uvicorn.run("vlm_server:app", host="0.0.0.0", port=args.port, workers=1, reload=False)