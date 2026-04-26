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
