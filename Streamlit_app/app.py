import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import textwrap
import torch
import os
import pandas as pd
import time
import plotly.express as px
from datetime import datetime
torch.cuda.set_device(6)

class RAGChatbot:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2', gpu_id=None):
        # Set GPU if specified
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = "cpu"
            
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Initialize embedding model
        self.embedder = SentenceTransformer(model_name)
        self.embedder.to(self.device)
        
        # Initialize Ollama
        self.llm = Ollama(model="llama3.3:70b-instruct-q8_0")
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Performance metrics
        self.metrics = {
            'embedding_time': [],
            'retrieval_time': [],
            'generation_time': [],
            'total_time': []
        }

    def get_relevant_context(self, query, k=6):
        start_time = time.time()
        
        # Create query embedding
        embed_start = time.time()
        query_embedding = self.embedder.encode([query], device=self.device)
        embed_time = time.time() - embed_start
        self.metrics['embedding_time'].append(embed_time)
        
        # Search in FAISS index
        retrieval_start = time.time()
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        retrieval_time = time.time() - retrieval_start
        self.metrics['retrieval_time'].append(retrieval_time)
        
        # Get relevant texts and their metadata
        context = []
        context_metadata = []
        
        for idx in indices[0]:
            meta = self.metadata[idx]
            context_metadata.append(meta)

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
            
        return context, context_metadata
    
    def generate_response(self, query, context):
        generation_start = time.time()
        
        # Create prompt with conversation history
        conversation_context = "\n".join([
            f"Human: {exchange['query']}\nAssistant: {exchange['response']}"
            for exchange in self.conversation_history[-3:]  # Include last 3 exchanges
        ])
        
        prompt = f"""You are an expert assistant in Kerala Government Rules (like KSR, KFC, KTC, KSSR etc.).  
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
        
        generation_time = time.time() - generation_start
        self.metrics['generation_time'].append(generation_time)
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'context': context,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        total_time = generation_time + self.metrics['embedding_time'][-1] + self.metrics['retrieval_time'][-1]
        self.metrics['total_time'].append(total_time)
        
        return response
    
    def chat(self, query):
        start_time = time.time()
        
        # Handle conversation management commands
        if query.lower() == 'clear history':
            self.conversation_history = []
            return "Conversation history cleared."
            
        # Normal query processing
        context, context_metadata = self.get_relevant_context(query)
        response = self.generate_response(query, context)
        
        return response, context_metadata
    
    def get_performance_metrics(self):
        metrics_df = pd.DataFrame({
            'Embedding Time (s)': self.metrics['embedding_time'],
            'Retrieval Time (s)': self.metrics['retrieval_time'],
            'Generation Time (s)': self.metrics['generation_time'],
            'Total Time (s)': self.metrics['total_time']
        })
        
        if len(metrics_df) > 0:
            avg_metrics = {
                'Average Embedding Time': metrics_df['Embedding Time (s)'].mean(),
                'Average Retrieval Time': metrics_df['Retrieval Time (s)'].mean(),
                'Average Generation Time': metrics_df['Generation Time (s)'].mean(),
                'Average Total Time': metrics_df['Total Time (s)'].mean()
            }
        else:
            avg_metrics = {
                'Average Embedding Time': 0,
                'Average Retrieval Time': 0,
                'Average Generation Time': 0,
                'Average Total Time': 0
            }
            
        return metrics_df, avg_metrics

# Initialize session state
def init_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False
    if 'show_metrics' not in st.session_state:
        st.session_state.show_metrics = False

def main():
    st.set_page_config(
        page_title="Kerala Government Rules Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    index_path = st.sidebar.text_input(
        "FAISS Index Path", 
        value="/workspace/Extracted/Structured/Summary/Vector_DB/embeddings.faiss"
    )
    
    metadata_path = st.sidebar.text_input(
        "Metadata Path", 
        value="/workspace/Extracted/Structured/Summary/Vector_DB/metadata.json"
    )
    
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
        index=0
    )
    
    gpu_options = ["CPU"] + [f"GPU {i}" for i in range(torch.cuda.device_count())]
    gpu_selection = st.sidebar.selectbox("Device", options=gpu_options, index=0)
    gpu_id = None if gpu_selection == "CPU" else int(gpu_selection.split(" ")[1])
    
    k_value = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=20, value=6)
    
    if st.sidebar.button("Initialize Chatbot"):
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = RAGChatbot(
                index_path=index_path,
                metadata_path=metadata_path,
                model_name=embedding_model,
                gpu_id=gpu_id
            )
        st.sidebar.success("Chatbot initialized!")
    
    # Toggle switches
    st.sidebar.subheader("Display Options")
    st.session_state.show_sources = st.sidebar.checkbox("Show Sources", value=st.session_state.show_sources)
    st.session_state.show_metrics = st.sidebar.checkbox("Show Performance Metrics", value=st.session_state.show_metrics)
    
    # Clear conversation
    if st.sidebar.button("Clear Conversation"):
        if st.session_state.chatbot:
            st.session_state.chatbot.conversation_history = []
        st.session_state.messages = []
    
    # Main content
    st.title("Kerala Government Rules Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if enabled and available
            if message.get("sources") and st.session_state.show_sources and message["role"] == "assistant":
                with st.expander("View Sources"):
                    source_df = pd.DataFrame(message["sources"])
                    columns_to_display = [col for col in source_df.columns if col != 'Text' and not source_df[col].isna().all()]
                    st.dataframe(source_df[columns_to_display])
    
    # Performance metrics section
    if st.session_state.show_metrics and st.session_state.chatbot:
        metrics_df, avg_metrics = st.session_state.chatbot.get_performance_metrics()
        
        if len(metrics_df) > 0:
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(metrics_df.tail())
                
            with col2:
                for metric_name, value in avg_metrics.items():
                    st.metric(metric_name, f"{value:.4f}s")
                
            # Create performance chart
            if len(metrics_df) > 1:
                chart_data = metrics_df.copy()
                chart_data['Query Number'] = range(1, len(chart_data) + 1)
                fig = px.line(
                    chart_data,
                    x='Query Number',
                    y=['Embedding Time (s)', 'Retrieval Time (s)', 'Generation Time (s)', 'Total Time (s)'],
                    title='Response Time Breakdown'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # User input
    if prompt := st.chat_input("Ask about Kerala Government Rules..."):
        if not st.session_state.chatbot:
            st.error("Please initialize the chatbot first!")
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = st.session_state.chatbot.chat(prompt)
                st.markdown(response)
                
                # Store sources for display
                if st.session_state.show_sources:
                    with st.expander("View Sources"):
                        source_df = pd.DataFrame(sources)
                        columns_to_display = [col for col in source_df.columns if col != 'Text' and not source_df[col].isna().all()]
                        st.dataframe(source_df[columns_to_display])
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()