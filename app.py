import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="NotebookLM", page_icon="📚")

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "")

# --- 2. CUSTOM VISUAL FLAIR & BRANDING REMOVAL ---
design_and_style = """
<style>
    /* Hide Streamlit default menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Base app gradient background */
    .stApp {
        background: linear-gradient(135deg, #0A0A0A 0%, #1A1A3A 100%);
        color: #E0E0E0;
    }

    /* Style the sidebar with depth */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111122 0%, #000000 100%);
        border-right: 1px solid #333344;
        color: #FFFFFF;
    }
    
    /* Give headers a clean, futuristic font and glow */
    h1, h2, h3, .stHeader {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        color: #06B6D4 !important; /* Bright, crisp Cyan */
        text-shadow: 0px 2px 10px rgba(6, 182, 212, 0.3);
    }

    /* Custom Card Styling for Chat History */
    [data-testid="stChatInput"] {
        background: #111122;
        border-radius: 50px !important;
        border: 1px solid #333344;
        color: #E0E0E0 !important;
        padding-left: 20px;
    }
    
    .chat-user {
        background: rgba(109, 40, 217, 0.2);
        border: 1px solid #6D28D9;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        color: white;
    }
    
    .chat-assistant {
        background: rgba(6, 182, 212, 0.2);
        border: 1px solid #06B6D4;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        color: white;
    }
    /* Clean headers for chat messages */
    .msg-header {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 4px;
    }
    .user-header { color: #A78BFA; } /* Subtle purple */
    .bot-header { color: #67E8F9; } /* Subtle cyan */

    /* Bouncing Dots Typing Animation */
    .typing-container {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 5px;
        padding-top: 5px;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #06B6D4;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
        40% { transform: scale(1); opacity: 1; }
    }
</style>
"""
st.markdown(design_and_style, unsafe_allow_html=True)

# --- 3. APP HEADER ---
st.title("📚 NotebookLM")
st.write("Upload a document, and ask questions based strictly on its content.")

# Sidebar
with st.sidebar:
    st.header("My Documents")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. RAG PIPELINE: INGESTION ---
if uploaded_file is not None and api_key:
    if st.session_state.vector_store is None:
        with st.spinner("📚 Reading and understanding your document..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                splits = text_splitter.split_documents(docs)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(splits, embeddings)
                
                st.session_state.vector_store = vector_store
                os.unlink(tmp_file_path)
                
                st.success("Document ready! You can now ask questions.")
            
            except Exception as e:
                st.error("There was an issue reading this document. Please try another file.")

elif not api_key:
    st.error("API Key not found. Please check your .env file.")

# --- 5. CHAT INTERFACE ---
# Display chat history using our custom CSS cards
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="chat-user">'
            f'<div class="msg-header user-header">👤 You</div>'
            f'<div>{message["content"]}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-assistant">'
            f'<div class="msg-header bot-header">📚 Notebook Assistant</div>'
            f'<div>{message["content"]}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

# User Input
if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.vector_store is None:
        st.error("Please upload a document first.")
    else:
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f'<div class="chat-user">'
            f'<div class="msg-header user-header">👤 You</div>'
            f'<div>{prompt}</div>'
            f'</div>', 
            unsafe_allow_html=True
        )

        # Create a temporary empty container for the loading animation
        message_placeholder = st.empty()
        
        # Show the "Typing..." card
        message_placeholder.markdown(
            f'<div class="chat-assistant">'
            f'<div class="msg-header bot-header">📚 Notebook Assistant</div>'
            f'<div class="typing-container">'
            f'<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>'
            f'</div></div>', 
            unsafe_allow_html=True
        )

        try:
            llm = ChatGroq(
                api_key=api_key,
                model_name="llama-3.1-8b-instant", 
                temperature=0.0 
            )

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If the answer is not contained in the context, explicitly say 'I cannot answer this based on the provided document.' "
                "Do NOT use your general knowledge. "
                "\n\nContext: {context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Generate the response
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            # Instantly swap the typing animation with the actual answer
            message_placeholder.markdown(
                f'<div class="chat-assistant">'
                f'<div class="msg-header bot-header">📚 Notebook Assistant</div>'
                f'<div>{answer}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            # Save to session history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        except Exception as e:
            message_placeholder.empty() # Clear the typing dots if it fails
            st.error("I'm sorry, I ran into an issue connecting to the AI. Please try asking again.")