from langchain_community.vectorstores import Chroma
import os
from data_processing import load_documents, chunk_documents

# Try the new import first, fall back to old if not available
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Using new HuggingFace embeddings import")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("Using legacy HuggingFace embeddings import")


def create_embeddings_openai(chunks, persist_directory="./data/embeddings"):
    """Create and store embeddings for document chunks using OpenAI"""
    from langchain_community.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    return vectorstore


def create_embeddings_free(chunks, persist_directory="./data/embeddings"):
    """Create and store embeddings using free model"""
    # Use a free embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    return vectorstore


def load_embeddings_openai(persist_directory="./data/embeddings"):
    """Load existing embeddings (OpenAI version)"""
    from langchain_community.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


def load_embeddings_free(persist_directory="./data/embeddings"):
    """Load existing embeddings (free version)"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


def test_embeddings():
    """Test the embedding creation and retrieval"""
    print("Testing embedding creation and retrieval...")

    # Check if we already have embeddings
    if os.path.exists("./data/embeddings") and os.listdir("./data/embeddings"):
        print("Loading existing embeddings...")
        vectorstore = load_embeddings_free()
        print(f"✓ Embeddings loaded successfully!")
        print(f"✓ Collection contains {vectorstore._collection.count()} documents")
    else:
        print("Creating new embeddings...")

        # First load and chunk documents
        documents = load_documents("./data/banking_documents")
        chunks = chunk_documents(documents)
        print(f"✓ Processed {len(chunks)} document chunks")

        # Create embeddings (using free version for testing)
        vectorstore = create_embeddings_free(chunks)
        print(f"✓ Embeddings created and saved successfully!")
        print(f"✓ Collection contains {vectorstore._collection.count()} documents")

    # Test retrieval with some banking queries
    test_queries = [
        "savings account requirements",
        "home loan interest rate",
        "credit card benefits",
        "fraud protection services"
    ]

    print("\nTesting document retrieval:")
    print("-" * 40)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vectorstore.similarity_search(query, k=2)
        print(f"Found {len(results)} relevant documents")

        for i, doc in enumerate(results):
            preview = doc.page_content[:100].replace('\n', ' ') + "..." if len(
                doc.page_content) > 100 else doc.page_content
            print(f"  {i + 1}. {preview}")


if __name__ == "__main__":
    test_embeddings()