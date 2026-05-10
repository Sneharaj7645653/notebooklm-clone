# 📚 NotebookLM Clone: Professional RAG Pipeline

A high-performance, private version of Google NotebookLM. This application allows users to upload documents (PDF/TXT) and engage in a grounded conversation where the AI answers questions based **strictly** on the provided content using a Retrieval-Augmented Generation (RAG) architecture.

---

## 🚀 Project Overview

This project implements a complete end-to-end RAG pipeline using:

- **UI:** Streamlit (Clean, minimal interface)
- **Orchestration:** LangChain
- **LLM:** Groq (Llama-3.1-8b-instant) for lightning-fast inference
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

---

## ✅ Feature Compliance (Mapping to Requirements)

Below is a breakdown of how this project meets the specific assignment criteria:

### 1. Full RAG Pipeline Implementation

The application follows the industry-standard RAG flow:

- **Ingestion:** Uses `PyPDFLoader` and `TextLoader` to handle multiple file formats.
- **Chunking:** Implements a logical splitting strategy (see below).
- **Embedding:** Converts text into 384-dimensional vectors using HuggingFace's sentence transformers.
- **Storage:** Stores vectors in a local `FAISS` index for high-speed similarity search.
- **Retrieval:** Uses Similarity Search to fetch the top 4 most relevant context chunks.
- **Generation:** Feeds the context into Groq's Llama 3.1 model to produce a grounded response.

### 2. Chunking Strategy (Documented)

We utilize the **RecursiveCharacterTextSplitter**.

- **Strategy:** This strategy is chosen because it attempts to split text on logical boundaries (double newlines for paragraphs, then single newlines for sentences).
- **Parameters:** `chunk_size=1000` and `chunk_overlap=200`.
- **Benefit:** Maintaining a 200-character overlap ensures that semantic context isn't lost between chunks, which is vital for answering questions that might span across a split.

### 3. Vector Database

The project uses **FAISS** as the vector database. Unlike a simple keyword search, FAISS allows the system to understand the *meaning* of the user's question and find the most semantically similar sections of the document, even if the exact words don't match.

### 4. Answer Quality & Grounding (Anti-Hallucination)

To ensure the LLM does not answer from its general training data:

- A **Strict System Prompt** is used:
  > "If the answer is not contained in the context, explicitly say 'I cannot answer this based on the provided document.' Do NOT use your general knowledge."

- **Temperature is set to 0.0** to ensure deterministic, factual output rather than creative or "hallucinated" content.

### 5. Handling Unseen Documents

The system is entirely dynamic. It does not rely on pre-indexed data. Every time a new file is uploaded, the FAISS index is cleared and rebuilt from scratch for that specific document, allowing it to handle any PDF or TXT file instantly.

---

## 🛠️ Local Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd notebooklm_clone
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_gsk_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```plaintext
notebooklm_clone/
│
├── app.py               # Core application logic and Streamlit UI
├── requirements.txt     # List of required Python dependencies
├── .env                 # Stores the Groq API key (ignored by git)
├── README.md            # Project documentation
├── .gitignore           # To keep your .env out of GitHub
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Frontend/UI | Streamlit |
| LLM | Groq - Llama 3.1 |
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace Transformers |
| Document Parsing | PyPDFLoader / TextLoader |

---

## 🔒 Grounded Response Design

The application is intentionally designed to avoid hallucinations:

- Responses are generated **only** from retrieved document chunks.
- The system prompt strictly prevents external knowledge usage.
- Low temperature (`0.0`) ensures factual consistency.
- If information is unavailable, the model explicitly declines to answer.

---

## 📈 Future Improvements

Potential enhancements include:

- Multi-document querying
- Persistent vector storage
- Chat history memory
- OCR support for scanned PDFs
- Source citation highlighting
- Hybrid search (keyword + semantic)

---

## 📄 License

This project is intended for educational and demonstration purposes.