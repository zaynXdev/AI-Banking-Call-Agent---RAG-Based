from generator_free import create_rag_chain, generate_response
from retriever import retrieve_documents
from embedding_generator import load_embeddings_free


def test_interactive():
    """Interactive test that lets you ask banking questions"""
    print("🤖 Banking RAG System - Interactive Test")
    print("=" * 50)

    # Initialize the chain
    print("Initializing language model... (this may take a moment)")
    rag_chain = create_rag_chain()

    # Load embeddings and vector store
    print("Loading banking knowledge base...")
    vectorstore = load_embeddings_free()
    print("✅ System ready! Ask me banking questions!\n")

    while True:
        # Get user question
        question = input("💬 Ask a banking question (or type 'quit' to exit): ")

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not question.strip():
            continue

        print(f"\n🔍 Searching for: '{question}'")

        try:
            # Retrieve relevant documents
            relevant_docs = retrieve_documents(question, vectorstore, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            print(f"📄 Found {len(relevant_docs)} relevant banking documents")

            if relevant_docs:
                print("   Retrieved information about:")
                for i, doc in enumerate(relevant_docs):
                    preview = doc.page_content[:100].replace('\n', ' ') + "..."
                    print(f"   {i + 1}. {preview}")

            if not context.strip():
                print("❌ No relevant information found in documents.")
                continue

            # Generate response
            print("\n💭 Generating answer...")
            response = generate_response(question, context, rag_chain)
            print(f"\n✅ Answer: {response}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    test_interactive()