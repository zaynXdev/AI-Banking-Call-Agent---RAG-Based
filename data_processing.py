import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(data_path):
    """Load banking documents from directory"""
    pdf_loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    text_loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={'autodetect_encoding': True}
    )

    documents = pdf_loader.load() + text_loader.load()
    return documents


def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    """Split documents into manageable chunks with better settings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_sample_banking_data():
    """Create better sample banking data with more content"""
    sample_content = """Savings Account Information

Opening a Savings Account:
To open a savings account, you need to provide valid government-issued ID, proof of address, and an initial deposit of at least $100. Our savings accounts offer competitive interest rates starting at 2.5% per annum with no monthly maintenance fees.

Savings Account Features:
- Minimum balance: $100
- Interest rate: 2.5% per annum
- Free online banking and mobile app
- Monthly statements provided
- No monthly maintenance fee
- ATM card included
- Mobile check deposit available
- 24/7 customer support

Home Loan Services

Home Loan Application Process:
Applying for a home loan requires documentation of income, credit history, and property details. Our mortgage specialists will guide you through the process from pre-approval to closing.

Home Loan Options:
- Interest rates from 5.5% APR
- Loan terms: 15, 20, or 30 years
- Maximum loan: 80% of property value
- Processing fee: 1% of loan amount
- Fixed and adjustable rates available
- Pre-approval within 24 hours
- Refinancing options

Credit Card Benefits

Premium Credit Card Features:
Our credit cards offer exceptional benefits including cashback rewards, travel insurance, and purchase protection. Choose from various cards tailored to your spending habits.

Credit Card Details:
- Annual fee: $50 (waived first year)
- Interest rate: 18% per annum
- Reward points on all purchases
- Contactless payment technology
- Cashback on groceries and gas
- Travel accident insurance
- Fraud protection guarantee
- No foreign transaction fees

Fraud Protection Services

24/7 Security Monitoring:
We employ advanced fraud detection systems that monitor your accounts around the clock. You'll receive instant alerts for suspicious activity.

Fraud Protection Features:
- Real-time transaction monitoring
- Zero liability protection
- Instant card blocking via mobile app
- SMS and email alerts for all transactions
- Two-factor authentication
- 256-bit encryption technology
- Regular security updates
- Dedicated fraud resolution team

Customer Service Support

Banking Assistance:
Our customer service team is available 24/7 to assist with any banking needs. Visit any branch, call our hotline, or use our mobile app for immediate support.

Contact Information:
- Phone support: 1-800-BANK-123 (24/7)
- Email support: support@bank.com (response within 2 hours)
- Branch locations: 150+ nationwide
- Mobile app: Available for iOS and Android
- Live chat: On our website and mobile app
- Social media: @BankSupport on Twitter and Facebook"""

    os.makedirs("./data/banking_documents", exist_ok=True)
    with open("./data/banking_documents/banking_services_detailed.txt", "w", encoding='utf-8') as f:
        f.write(sample_content)

    print("Detailed banking data created!")


# Test the functions
if __name__ == "__main__":
    # Create better sample data
    print("Creating improved banking documents...")
    create_sample_banking_data()

    # Test with new documents
    documents = load_documents("./data/banking_documents")
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Show sample chunks
    if chunks:
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i + 1} ---")
            print(chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content)