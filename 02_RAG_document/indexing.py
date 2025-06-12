from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # Vector Embeddings banane ke liye isse import kiya hai
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Loading
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load() # Read PDF file
# print("Docs[0]: ", docs[5])


# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
splitted_docs = text_splitter.split_documents(documents=docs)


# Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Using embedding_model create embeddings of splitted_docs and store in DB
vector_store = QdrantVectorStore.from_documents(
    documents=splitted_docs,
    url="http://vector-db:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)
# print("indexing of documents done...")
