import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import os
import anthropic

class RAGChatbot:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Initialize embedding model
        self.embedder = SentenceTransformer(model_name)
        
        # Initialize Anthropic API
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
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

            context_string = ', '.join(fields)
            context.append(context_string)
        return context
    
    def generate_response(self, query, context):
        # Create prompt with conversation history
        conversation_context = "\n".join([
            f"Human: {exchange['query']}\nAssistant: {exchange['response']}"
            for exchange in self.conversation_history[-3:]  # Include last 3 exchanges
        ])
        
        prompt = f"""You are an expert assistant in Kerala Government Rules like KSR, KFC, KTC, KSSR etc.  
Follow these guidelines for your responses:
1. Use simple, everyday language that anyone can understand.
2. Organize your answer in clear paragraphs with one main idea per paragraph.
3. Start with the most important information first.
4. Include proper references (document, part, chapter, rule number, etc.) when available.
5. Clearly state if the answer cannot be found in the provided rules.
6. Avoid technical jargon unless absolutely necessary, and explain any technical terms you must use.
7. Use short sentences and simple sentence structure.
8. DO NOT fabricate information. If the answer is not found in the rules, explicitly state so.

Previous conversation:
{conversation_context}

Relevant Rules:
{' '.join(context)}

Current question: {query}

Answer:"""
        
        # Generate response using Anthropic API
        response = self.client.messages.create(
            model="claude-3-7-sonnet-latest",  # Using Claude 3 Sonnet
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': response_text,
            'context': context
        })
        
        return response_text
    
    def chat(self, query):
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
        '/workspace/Extracted/Structured/Summary/Vector_DB/metadata.json'
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

if __name__ == "__main__":
    main()
