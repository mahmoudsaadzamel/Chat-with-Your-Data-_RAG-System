# main.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
import hashlib

import os
# set the API key 
OPENAI_API_KEY = "OPENAI_API_KEY"
# Paths
FAISS_INDEX_PATH = "vectorstore_index"
cache = {}  # Use a Python dictionary for caching


def get_loader(file_path):
    file_path = file_path.strip()  # Remove extra spaces
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[-1].lower().lstrip(".")  # Extract extension

    if not file_extension:
        raise ValueError(f"Error: File has no extension! Please rename it correctly: {file_path}")

    if file_extension == "pdf":
        return PyMuPDFLoader(file_path)
    elif file_extension == "txt":
        return TextLoader(file_path)
    elif file_extension == "csv":
        return CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format! File provided: {file_path}")

#Function to process the file and create vectorstore
def process_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found - {file_path}")

    print(f"Processing file: {file_path}")  # Debugging
    file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

    if file_hash in cache:  
        print("Loading cached vectorstore...")
        return cache[file_hash]

    try:
        loader = get_loader(file_path)  # Ensure file is correctly detected
        documents = loader.load()
        print(f"{file_path} loaded successfully!")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)

        vectorstore.save_local(FAISS_INDEX_PATH)
        cache[file_hash] = vectorstore  # Store in dictionary cache

        return vectorstore

    except Exception as e:
        print(f"Error processing file: {e}")
        raise e


# Function to load stored vectorstore
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading saved Vector Store...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# Function to query the RAG system
def query_rag_system(query):
    if qa_chain is None:
        raise ValueError("QA system is not initialized!")
    
    result = qa_chain.run(query)
    return result


# Function to create the QA system
def create_qa_system(vectorstore):
    print("Creating QA system...")

    # Custom prompt template
    prompt_template = """
    You are an AI assistant. Use the following pieces of context to answer the user's question.
    If no context is available, answer politely using your general knowledge.

    Context:
    {context}

    Question: {question}

    Only provide the helpful answer below.
    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )

    print("QA system ready!")
    return qa_chain


# Initialize vectorstore
vectorstore = load_vectorstore() if load_vectorstore() else None

# Create QA system if vectorstore is available
qa_chain = create_qa_system(vectorstore) if vectorstore else None