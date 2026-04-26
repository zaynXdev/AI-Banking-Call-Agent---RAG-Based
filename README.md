# 🏦 AI Banking Call Agent (RAG-Based)

An intelligent **AI-powered banking assistant** built using a **Retrieval-Augmented Generation (RAG)** pipeline.  
This system answers banking-related queries by retrieving relevant information from custom documents and generating accurate, context-aware responses using LLMs.

---

## 🚀 Overview

This project simulates a **banking call agent** that can:
- Answer customer queries about banking services
- Retrieve information from structured/unstructured documents
- Generate reliable responses grounded in real data

Unlike traditional chatbots, this system uses **RAG**, which improves accuracy by combining retrieval + generation instead of relying only on model knowledge :contentReference[oaicite:0]{index=0}.

---

## 🧠 Key Features

- 🔍 **Document-Based Q&A**  
  Answers are generated using banking documents (PDF/TXT)

- 🧾 **RAG Pipeline**
  - Document loading & preprocessing
  - Embedding generation
  - Vector-based retrieval
  - LLM response generation

- ⚡ **Context-Aware Responses**  
  Retrieves relevant chunks before generating answers

- 🧩 **Modular Architecture**  
  Separate components for:
  - Embeddings
  - Retrieval
  - Generation

- 🧪 **Interactive Testing Script**  
  Test queries using a CLI-based interface

---

## 🏗️ Project Structure
RAG_for_banking_system/
│
├── data/
│ ├── banking_documents/
│ │ ├── banking_services_detailed.txt
│ │ └── sample_banking.pdf
│ └── embeddings/
│
├── data_processing.py # Document preprocessing
├── embedding_generator.py # Generates embeddings
├── retriever.py # Retrieves relevant context
├── generator_free.py # LLM response generation
├── test_interactive.py # CLI testing interface
├── err.py # Error handling utilities
├── .env # API keys (not included in repo)


---

## ⚙️ How It Works

1. **Load Documents**  
   Banking documents are processed and split into chunks

2. **Generate Embeddings**  
   Each chunk is converted into vector embeddings

3. **Store Embeddings**  
   Stored locally for fast retrieval

4. **User Query → Retrieval**  
   Relevant chunks are retrieved using similarity search

5. **LLM Generation**  
   Retrieved context + query → final response

This approach reduces hallucination and improves factual accuracy in AI systems :contentReference[oaicite:1]{index=1}.

---

## 🛠️ Tech Stack

- **Python**
- **LLMs (OpenAI / Free APIs)**
- **Vector Embeddings**
- **RAG Architecture**
- **Environment Variables (.env)**

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/zaynXdev/AI-Banking-Call-Agent---RAG-Based.git

cd AI-Banking-Call-Agent---RAG-Based

# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

🔑 Environment Setup
Create a .env file:
OPENAI_API_KEY=your_api_key_here
⚠️ Never commit your .env file.

▶️ Run the Project
Step 1: Generate embeddings
python embedding_generator.py
Step 2: Run interactive agent
python test_interactive.py
 
