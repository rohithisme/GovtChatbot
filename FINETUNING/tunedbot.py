import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

class RAGChatbot:
    def __init__(self, index_path, metadata_path, 
                 llama_model_path="/workspace/FINETUNING/llama-merged",
                 embedding_model_name='all-MiniLM-L6-v2'):
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Initialize fine-tuned Llama model
        print("Loading fine-tuned Llama model...")
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")
        
        # Initialize conversation history
        self.conversation_history = []

    def get_relevant_context(self, query, k=6):
        # Create query embedding
        query_embedding = self.embedder.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get relevant texts and their metadata
        context = []
        for idx in indices[0]:
            meta = self.metadata[idx]

            # Create a list of field-value pairs, excluding empty values
            fields = []
            if meta.get('Document'):
                fields.append(f"Document: {meta['Document']}")
            if meta.get('Part'):
                fields.append(f"Part: {meta['Part']}")
            if meta.get('Chapter'):
                fields.append(f"Chapter: {meta['Chapter']}")
            if meta.get('Appendix'):
                fields.append(f"Appendix: {meta['Appendix']}")
            if meta.get('Annexure'):
                fields.append(f"Annexure: {meta['Annexure']}")
            if meta.get('Section'):
                fields.append(f"Section: {meta['Section']}")
            if meta.get('Sub Section'):
                fields.append(f"Sub Section: {meta['Sub Section']}")
            if meta.get('Sub division'):
                fields.append(f"Sub division: {meta['Sub division']}")
            if meta.get('Rule no.'):
                fields.append(f"Rule: {meta['Rule no.']}")
            if meta.get('Amendment order no.'):
                fields.append(f"Amendment Order: {meta['Amendment order no.']}")
            if meta.get('Order date'):
                fields.append(f"Order Date: {meta['Order date']}")
            if meta.get('Effective date'):
                fields.append(f"Effective Date: {meta['Effective date']}")
            if meta.get('Description'):
                fields.append(f"Description: {meta['Description']}")

            # Join all non-empty fields with commas
            context_string = ', '.join([f for f in fields if f])
            context.append(context_string)
        return context
    
    def clean_response(self, response):
        """Clean up the model response to extract only the actual answer."""
        # Pattern to match the assistant's answer at the end of the response
        assistant_pattern = r"Answer:assistant\s*(.*?)(?:\n\nYou:|$)"
        match = re.search(assistant_pattern, response, re.DOTALL)
        
        if match:
            # Extract just the assistant's answer
            return match.group(1).strip()
        
        # Alternative pattern if the first one doesn't match
        alt_pattern = r"assistant\s*(.*?)(?:\n\nYou:|$)"
        match = re.search(alt_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
            
        # If both patterns fail, return the original response trimmed of common prefixes
        cleaned = response
        prefixes_to_remove = [
            "system", "user", "assistant", 
            "Answer:", "Generating response..."
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                
        return cleaned
    
    def generate_response(self, query, context):
        # Create prompt with conversation history
        conversation_context = "\n".join([
            f"Human: {exchange['query']}\nAssistant: {exchange['response']}"
            for exchange in self.conversation_history[-3:]  # Include last 3 exchanges
        ])
        
        prompt = f"""You are an expert assistant in Kerala Service Rules (KSR).  
Follow these guidelines for your responses:
1. Use simple, everyday language that anyone can understand
2. Organize your answer in clear paragraphs with one main idea per paragraph
3. Start with the most important information first
4. Include proper references (document, part, chapter, rule number, etc.) when available
5. Clearly state if the answer cannot be found in the provided rules
6. Avoid technical jargon unless absolutely necessary, and explain any technical terms you must use
7. Use short sentences and simple sentence structure
8. DO NOT fabricate information. If the answer is not found in the rules, explicitly state so.

Previous conversation:
{conversation_context}

Relevant Rules:
{' '.join(context)}

Current question: {query}

Answer:"""

        # Create the chat format for Llama model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Format for Llama 3.1
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate a response
        print("Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response to extract just the answer
        model_response = self.clean_response(full_response)
        
        # For debugging
        if not model_response or len(model_response) < 10:
            print("\nWarning: Extracted response may be incomplete")
            print("Full response:", full_response)
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': model_response,
            'context': context
        })
        
        return model_response
    
    def chat(self, query):
        # Handle conversation management commands
        if query.lower() == 'clear history':
            self.conversation_history = []
            return "Conversation history cleared."
            
        if query.lower() == 'show history':
            history = "\n\n".join([
                f"Human: {exchange['query']}\nAssistant: {exchange['response']}"
                for exchange in self.conversation_history
            ])
            return f"Conversation History:\n{history}"
        
        # Normal query processing
        context = self.get_relevant_context(query)
        response = self.generate_response(query, context)
        return response

def main():
    # Initialize chatbot
    chatbot = RAGChatbot(
        '/workspace/Extracted/Structured/Summary/Vector_DB/embeddings.faiss', 
        '/workspace/Extracted/Structured/Summary/Vector_DB/metadata.json',
        llama_model_path="/workspace/FINETUNING/llama-merged"  # Path to your fine-tuned model
    )
    
    print("KSR Chatbot initialized. Commands:")
    print("- Type 'quit' to exit")
    print("- Type 'clear history' to clear conversation history")
    print("- Type 'show history' to view conversation history")
    
    while True:
        query = input("\nYou: ").strip()
        
        if query.lower() == 'quit':
            print("\nGoodbye!")
            break
            
        try:
            response = chatbot.chat(query)
            print("\nAssistant:", textwrap.fill(response, width=80))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()