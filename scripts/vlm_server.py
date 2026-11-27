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
TORCH_DTYPE = torch.float16 # 明确使用 torch.float16，如果GPU支持且希望节省显存。
                           # 或者 torch.bfloat16 (适用于 Ampere 及更新架构的GPU)
                           # 或者 "auto" (让 Transformers 自动选择)
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
    temperature: Optional[float] = None # 新增
    do_sample: Optional[bool] = None   # 新增

class VLMTInferResponse(BaseModel):
    generated_text: str

# --- FastAPI Event Handling: Load Model and Processor on Startup ---
@app.on_event("startup")
async def load_resources():
    global model, processor
    print(f"开始加载 Processor: {MODEL_PATH}")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        print("Processor 加载成功！")
    except Exception as e:
        print(f"加载 Processor 失败: {e}")
        raise RuntimeError(f"无法加载 Processor: {e}")

    print(f"开始加载模型: {MODEL_PATH}")
    try:
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": TORCH_DTYPE, # 使用上面定义的 TORCH_DTYPE
            "device_map": DEVICE_MAP,
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
            # "quantization_config": quant_config  # 移除了量化配置
        }
        if ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
            print(f"使用 attn_implementation: {ATTN_IMPLEMENTATION}")

        print(f"模型加载参数: {model_kwargs}") # 打印模型加载参数以供调试

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            **model_kwargs
        )
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc() # 打印完整的堆栈跟踪
        raise RuntimeError(f"无法加载模型: {e}")

# --- API Endpoint for Inference ---
@app.post("/infer", response_model=VLMTInferResponse)
async def infer(request: VLMTInferRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型或Processor正在加载或加载失败，请稍后再试。")

    print(f"收到推理请求: max_new_tokens='{request.max_new_tokens}'")

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
            # 对于 device_map="auto"，HF `generate` 通常会把输入放到模型第一块所在的设备
            # 但为了保险，可以这样做，或者干脆不移动，让 `generate` 处理
             inputs = inputs.to(model.device) # model.device 会指向模型第一个参数所在的设备
        elif torch.cuda.is_available():
             inputs = inputs.to("cuda")
        else:
             inputs = inputs.to("cpu")


        print("正在生成文本...")
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

        print(f"生成文本: {output_text}")
        return VLMTInferResponse(generated_text=output_text)

    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000) # Default port, can be changed
    args = parser.parse_args()
    
    uvicorn.run("vlm_server:app", host="0.0.0.0", port=args.port, workers=1, reload=False)