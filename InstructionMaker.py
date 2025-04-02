import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import torch
import os
from tqdm import tqdm
import time

torch.cuda.set_device(6)

class RAGChatbot:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2', checkpoint_dir='checkpoints'):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.embedder = SentenceTransformer(model_name)
        self.llm = Ollama(model="llama3.3:70b-instruct-q8_0")
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def get_relevant_context(self, query, k=6):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
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

    def generate_response(self, query, context):
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

        Relevant Rules:
        {' '.join(context)}

        Question: {query}
        Answer:"""
        
        return self.llm.invoke(prompt)
    
    def save_checkpoint(self, results, checkpoint_name):
        """Save progress checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nCheckpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name):
        """Load progress from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return None

    def process_questions(self, question_file, output_file, checkpoint_size=20):
        # Load questions
        with open(question_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        # Check for most recent checkpoint
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.json')]
        most_recent_checkpoint = None
        start_idx = 0
        results = []
        
        if checkpoints:
            # Sort checkpoints by their timestamp
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            most_recent_checkpoint = checkpoints[-1]
            loaded_results = self.load_checkpoint(most_recent_checkpoint)
            
            if loaded_results:
                results = loaded_results
                start_idx = len(results)
                print(f"Resuming from checkpoint with {start_idx} questions already processed")
        
        # Process remaining questions with progress bar
        if start_idx < len(questions):
            print(f"Processing questions {start_idx+1} to {len(questions)}")
            
            for i in tqdm(range(start_idx, len(questions)), desc="Processing questions"):
                question = questions[i]
                try:
                    context = self.get_relevant_context(question)
                    response = self.generate_response(question, context)
                    results.append({"instruction": question, "input": "", "output": response})
                    
                    # Create checkpoint after every checkpoint_size questions
                    if (i + 1) % checkpoint_size == 0 or i == len(questions) - 1:
                        checkpoint_name = f"checkpoint_{len(results)}.json"
                        self.save_checkpoint(results, checkpoint_name)
                        
                        # Remove older checkpoints, keeping only the 3 most recent
                        self._cleanup_old_checkpoints(3)
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"\nError processing question {i+1}: {e}")
                    # Save checkpoint on error
                    checkpoint_name = f"checkpoint_{len(results)}_error.json"
                    self.save_checkpoint(results, checkpoint_name)
                    raise
        
        # Save final output
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\nAll responses saved to {output_file}")
    
    def _cleanup_old_checkpoints(self, keep_count=3):
        """Keep only the most recent checkpoints."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.json')]
        
        # Sort checkpoints by their timestamp (numeric part)
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Remove older checkpoints, leaving the most recent ones
        if len(checkpoints) > keep_count:
            for old_checkpoint in checkpoints[:-keep_count]:
                try:
                    os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
                except Exception as e:
                    print(f"Error removing old checkpoint {old_checkpoint}: {e}")

if __name__ == "__main__":
    chatbot = RAGChatbot(
        '/workspace/Extracted/Structured/Summary/Vector_DB/embeddings.faiss',
        '/workspace/Extracted/Structured/Summary/Vector_DB/metadata.json'
    )
    chatbot.process_questions('/workspace/Questions.txt', '/workspace/Responses.json')