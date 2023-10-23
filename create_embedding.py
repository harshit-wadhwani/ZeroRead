from langchain.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from dotenv import load_dotenv
from pathlib import Path 

env_path =  Path('.') / '.env'
load_dotenv(dotenv_path= env_path)

def embedding_creation(filename, subjectname):
    persist_directory = f"./{subjectname}"
    pdf_path = filename
    embeddings = GooglePalmEmbeddings()
    loader = UnstructuredPDFLoader(pdf_path, strategy = 'hi_res')
    documents = loader.load()
    print("starting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    print("starting...")
    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    print('here')
    vectordb.persist()
    print("successfull")
    
embedding_creation("path_to_book", "name_of_embedding")