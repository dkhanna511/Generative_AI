# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama  ## This would be for opensource LLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import OpenLLM


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv() ## I have one file called .env. This has the API KEY Mentioned. this line loads the .env file
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY_MAIN") ## This takes in the Langchain api key from the .env file
# os.environ = "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"


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
    
)


##streamlit framework
st.title('LangChain Demo with Gemma (Ollama)')
input_text = st.text_input("Search the topic you want")



## Calling hunggingface llm
llm = Ollama(model = "gemma3:1b")


output_parser = StrOutputParser()
chain = prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))

