from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Path to your fine-tuned model
model_path = "/workspace/FINETUNING/llama-merged"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False  # Try using the slower but more accurate tokenizer
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_response(prompt, max_new_tokens=512, temperature=0.8):
    """
    Generate a Malayalam response from the model with fixes for encoding issues.
    """
    # Create messages with proper encoding
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Format prompt
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error in chat template: {e}")
        # Fallback to simple prompt
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response with adjusted parameters for Malayalam text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            # Removed early_stopping to avoid warning
            num_return_sequences=1
        )
    
    # Decode with proper handling for Malayalam characters
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    # Method 1: Look for standard assistant markers
    assistant_markers = [
        "\nassistant", "[/INST]", "</s>", "<assistant>",
        "ഉത്തരം:", "ഉത്തരം :"  # Malayalam answer markers
    ]
    
    response = full_response
    
    # Try to extract just the assistant's response
    for marker in assistant_markers:
        if marker in full_response:
            parts = full_response.split(marker, 1)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    # Remove any system and user parts that might be in the response
    system_user_patterns = [
        r"system.*?user", 
        r"user.*?assistant",
        r"Cutting Knowledge Date:.*?user",
        r"Today Date:.*?user"
    ]
    
    for pattern in system_user_patterns:
        response = re.sub(pattern, "", response, flags=re.DOTALL)
    
    # Clean up any special tokens that might have leaked through
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<s>", "</s>"]
    for token in special_tokens:
        response = response.replace(token, "")
    
    response = response.strip()
    
    # If response still contains the prompt, remove it
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    return response

def interactive_chat():
    print("Kerala Government Rules Assistant (മലയാളം) - Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        user_input = input("\nചോദ്യം: ")  # "Question" in Malayalam
        if user_input.lower() in ["exit", "quit", "പുറത്തുകടക്കുക"]:
            print("വിട! (Goodbye!)")
            break
            
        print("ഉത്തരം തയ്യാറാക്കുന്നു... (Generating response...)")
        
        try:
            response = generate_response(user_input)
            print(f"\nഉത്തരം: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Please try a different question or restart the application.")

# For debugging and fixing encoding issues
def debug_encoding(text):
    """Print details about a string's encoding"""
    print(f"Text length: {len(text)}")
    print(f"Encoded in UTF-8: {text.encode('utf-8')}")
    print(f"Unicode code points:", [ord(c) for c in text[:20]])  # First 20 chars

if __name__ == "__main__":
    # You can uncomment this to check the model's raw output for debugging
    # debug_sample = "ഒരു ഡിപ്പാർട്ട്‌മെൻ്റൽ കൈമാറ്റം എങ്ങനെ നടത്താം?"
    # raw_output = generate_response(debug_sample)
    # print("\nRaw model output:")
    # print(raw_output)
    # debug_encoding(raw_output)
    
    interactive_chat()