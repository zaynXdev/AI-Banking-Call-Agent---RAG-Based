from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def retrieve_documents(query, vectorstore, k=3):
    """Retrieve relevant documents for a query"""
    results = vectorstore.similarity_search(query, k=k)
    return results


def test_retrieval():
    """Test the retrieval system with banking queries"""
    print("Testing banking document retrieval...")

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./data/embeddings",
        embedding_function=embeddings
    )

    # Test specific banking queries
    test_queries = [
        "What are the requirements to open a savings account?",
        "What is the interest rate for home loans?",
        "What benefits does the credit card offer?",
        "How does fraud protection work?",
        "What is the minimum balance for savings?"
    ]

    print("Banking Query Results:")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve_documents(query, vectorstore, k=2)

        for i, doc in enumerate(results):
            content_preview = doc.page_content.replace('\n', ' ')[:100] + "..."
            print(f"  Result {i + 1}: {content_preview}")

        print("-" * 30)


if __name__ == "__main__":
    test_retrieval()