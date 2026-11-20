from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the PDF document

loader = PyPDFLoader("data/company_policy.pdf")  # Replace with your PDF file path
documents = loader.load()

# 2. Split the document into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(documents)
print(f"Number of chunks created: {len(chunks)}")

# 3. Create embeddings for the chunks

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vectorstore DB
vectorstore = FAISS.from_documents(
    chunks, embeddings
)

# Save the Vector store DB
vectorstore.save_local("vectorstore") 
print("Vector store saved locally.")