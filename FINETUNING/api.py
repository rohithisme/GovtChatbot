import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

app = FastAPI()

# Load the model and tokenizer
model_path = "/workspace/FINETUNING/llama-merged"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

@app.post("/chat", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    try:
        # Prepare the input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate response
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()