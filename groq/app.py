import streamlit as st
import os
from langchain_groq import ChatGroq

## This is all for document chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from langchain_community.vectorstores import FAISS

## this of for retriever chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time
load_dotenv()

## load the Groq API Key
groq_api_key = os.environ["GROQ_LPU_API_KEY"]

# print("Groq API Key:", groq_api_key)

# exit()
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model = "all-minilm")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.1-8b-instant")
response = llm.invoke("What is Groq?")
print(response)
exit()
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most
    accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Response time : ", time.process_time() - start)
    st.write(response['answer'])
    
    # Write a streamlit expander
    with st.expander("Document Similarity Search"):
        ## Find the relecant chunksChat
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------")

