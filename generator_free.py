from transformers import pipeline
import random


def setup_llm_free():
    """Initialize with a small, fast model for question answering"""
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",  # Small, fast, and good for QA
        max_length=200,
        temperature=0.3,
        do_sample=True,
        truncation=True
    )
    return llm


def create_rag_chain():
    """Create a RAG chain optimized for banking questions"""

    def simple_chain(inputs):
        context = inputs["context"]
        question = inputs["question"]

        # Clean and concise prompt
        prompt = f"""Based on the banking context below, answer this question:

Context: {context}

Question: {question}

Answer:"""

        try:
            llm = setup_llm_free()
            response = llm(
                prompt,
                max_length=150,
                num_return_sequences=1,
                early_stopping=True
            )[0]['generated_text']

            return response.strip()

        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"

    return simple_chain


def generate_response(question, context, chain):
    """Generate response using the RAG chain"""
    response = chain({"question": question, "context": context})
    return response