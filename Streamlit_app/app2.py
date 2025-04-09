import streamlit as st 
from PIL import Image
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import torch
torch.cuda.set_device(6)

# Set page config
st.set_page_config("Kerala Rules Chatbot", layout="centered")

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 150px;
        max-width: 200px;
    }
    [data-testid="stSidebar"] button {
        width: 100%;
        padding: 5px;
    }
    .main .block-container {
        max-width: 900px;
        padding-top: 0rem;
        padding-right: 4rem;
        padding-left: 2rem;
        margin: 0 auto;
        padding-bottom: 120px;  /* Ensure space at the bottom for input box */
    }

    /* Fix input to bottom */
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem 2rem;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 100;
    }

    .stTextInput {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Load logo
logo = Image.open('/workspace/Streamlit_app/Government_of_Kerala_Logo.png')  # Update path as needed
col1, col2, col3 = st.columns([2, 3, 1])
with col2:
    st.image(logo, width=150)

# Title
st.title("🧑‍⚖️ Kerala Govt Rules Chatbot")

# ---- RAG Chatbot ---- #
class RAGChatbot:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.embedder = SentenceTransformer(model_name)
        self.llm = Ollama(model="llama3.3:70b-instruct-q8_0")

    def get_relevant_context(self, query, k=6):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        context = []
        for idx in indices[0]:
            meta = self.metadata[idx]
            fields = []
            for key in ["Document", "Part", "Chapter", "Appendix", "Annexure",
                        "Section", "Sub Section", "Sub division", "Rule no.",
                        "Amendment order no.", "Order date", "Effective date", "Description"]:
                if meta.get(key):
                    label = key.replace("no.", "").replace("Sub ", "Sub-")
                    fields.append(f"{label}: {meta[key]}")
            context.append(', '.join(fields))
        return context

# ---- Initialize Chatbot ---- #
@st.cache_resource
def init_bot():
    return RAGChatbot(
        '/workspace/Extracted/Structured/Summary/Vector_DB/embeddings.faiss',
        '/workspace/Extracted/Structured/Summary/Vector_DB/metadata.json'
    )

bot = init_bot()

# ---- Session State ---- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of dicts: {'query', 'response', 'context'}

with st.container():
    st.markdown("Ask me about **KSR, KFC, KTC, KSSR**, and more!")

# ---- Display Chat History ---- #
for msg in st.session_state.messages:
    st.markdown(f"**🧍 You:** {msg['query']}")
    if msg["response"]:
        st.markdown(f"**🤖 Assistant:** {msg['response']}")
    else:
        st.markdown("**🤖 Assistant:** Thinking... 💭")
    st.markdown("---")

# ---- User Input (Fixed Bottom) ---- #
chat_input_container = st.container()
with chat_input_container:
    with st.form("chat_form", clear_on_submit=True):
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        user_input = st.text_input("Type your question here...", placeholder="Ask the assistant")
        submitted = st.form_submit_button("Send")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted and user_input:
        st.session_state.messages.append({
            "query": user_input,
            "response": "",
            "context": []
        })
        st.rerun()

# ---- Generate Assistant Response With Streaming ---- #
if st.session_state.messages and st.session_state.messages[-1]["response"] == "":
    try:
        last_msg = st.session_state.messages[-1]
        context = bot.get_relevant_context(last_msg["query"])
        conversation_history = st.session_state.messages[:-1]

        conversation_context = "\n".join([
            f"Human: {ex['query']}\nAssistant: {ex['response']}"
            for ex in conversation_history[-3:]
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
9. If the current question is unrelated to the previous conversation, completely ignore the conversation history and answer only based on the current question and relevant rules.

Previous conversation:
{conversation_context}

Relevant Rules:
{' '.join(context)}

Current question: {last_msg['query']}

Answer:"""

        # Streaming output
        response_box = st.empty()
        streamed_text = ""
        for chunk in bot.llm.stream(prompt):
            streamed_text += chunk
            response_box.markdown(f"**🤖 Assistant:** {streamed_text}▌")

        # Save result
        st.session_state.messages[-1]["response"] = streamed_text
        st.session_state.messages[-1]["context"] = context
        st.rerun()

    except Exception as e:
        st.session_state.messages[-1]["response"] = f"Error generating response: {e}"
        st.rerun()

# ---- Reset Chat ---- #
with st.sidebar:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
