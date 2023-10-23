import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import os, glob

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(page_title="ZeroRead", layout="wide",page_icon="📚",initial_sidebar_state="expanded")

prompt_template  = """
Use the following piece of context to answer the question. Please provide a detailed response, that should be long for each of the question. Provide formula or code for whichever question possible.

{context}

Question: {question}
"""

prompt = PromptTemplate(template = prompt_template , input_variables=["context", "question"])

# Sidebar contents
with st.sidebar:
    st.title("Built by - [Harshit Wadhwani](https://www.linkedin.com/in/harshitwadhwani/)")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built By Harshit Wadhwani:
    - [Linkedin](https://www.linkedin.com/in/harshitwadhwani/)
    - [Email ](mailto:harshit4work@gmail.com)
    - [Github](https://github.com/harshit-wadhwani) 
 
    """
    )
    add_vertical_space(5)
    

load_dotenv()

def main():
    st.header("ZeroRead 📚💬")
    st.write("Books that are used are exactly from our syllabus and are available [here](https://drive.google.com/drive/folders/1fbdrnSsf4zO2cI7R8mi57YuWVstkoIK5?usp=share_link)")
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

    select_subject = st.selectbox("Pick a Subject", ["18AI744-Business Intelligence", "18AI734-Cloud Computing"])
    
    embeddings=GooglePalmEmbeddings()

    vectordb = Chroma(persist_directory=select_subject, embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3,'include_metadata': True})

    llm = GooglePalm(temperature=0.5)  
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt}
    )

    query = st.text_input("Ask questions from the book:")
   
    if query:
        response = chain(query)
        st.write(response["result"])
        with st.expander("TextBook -  Context"):
            for doc in response["source_documents"]:
                st.write(f"{doc.metadata['source']} \n {doc.page_content}")


if __name__ == "__main__":
    main()
