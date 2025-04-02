import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import torch
import os

# Set page configuration
st.set_page_config(
    page_title="Kerala Service Rules Chatbot",
    page_icon="📚",
    layout="wide"
)

class RAGChatbot:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Initialize embedding model
        self.embedder = SentenceTransformer(model_name)
        
        # Initialize Ollama
        self.llm = Ollama(model="llama3.3:70b-instruct-q8_0")
        
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

        # Generate response using Ollama
        response = self.llm.invoke(prompt)
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'context': context
        })
        
        return response
    
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

# Initialize session state variables if they don't exist
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_chatbot():
    """Initialize the chatbot with FAISS index and metadata."""
    st.session_state.chatbot = RAGChatbot(
        index_path=st.session_state.index_path,
        metadata_path=st.session_state.metadata_path
    )
    st.session_state.initialized = True
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.title("KSR Chatbot Setup")
    
    # Configuration inputs
    st.header("Vector Database Paths")
    index_path = st.text_input(
        "FAISS Index Path", 
        value="/workspace/rohith_llm/Extracted/Structured/Summary/Vector_DB/embeddings.faiss",
        key="index_path"
    )
    metadata_path = st.text_input(
        "Metadata JSON Path", 
        value="/workspace/rohith_llm/Extracted/Structured/Summary/Vector_DB/metadata.json",
        key="metadata_path"
    )
    
    # GPU settings
    st.header("GPU Settings")
    use_gpu = st.checkbox("Use GPU", value=True)
    if use_gpu:
        gpu_id = st.number_input("GPU ID", min_value=0, max_value=8, value=6, step=1)
        if st.button("Set GPU"):
            torch.cuda.set_device(int(gpu_id))
            st.success(f"Set GPU to device {gpu_id}")
    
    # Initialize button
    if st.button("Initialize Chatbot"):
        with st.spinner("Initializing chatbot..."):
            initialize_chatbot()
        st.success("Chatbot initialized successfully!")
    
    # Actions
    st.header("Actions")
    if st.button("Clear Chat History") and st.session_state.initialized:
        st.session_state.messages = []
        st.session_state.chatbot.conversation_history = []
        st.success("Chat history cleared!")

# Main chat interface
st.title("Kerala Service Rules Chatbot")

# Check if chatbot is initialized
if not st.session_state.initialized:
    st.warning("Please initialize the chatbot using the sidebar controls before starting the conversation.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about Kerala Service Rules..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Documentation in the footer
st.markdown("---")
st.markdown("""
**Commands:**
- Type 'clear history' to reset the conversation
- Type 'show history' to view previous exchanges
""")
