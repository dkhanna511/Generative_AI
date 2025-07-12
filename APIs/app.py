from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

from dotenv import load_dotenv

load_dotenv() ## I have one file called .env. This has the API KEY Mentioned. this line loads the .env file

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY_MAIN") ## This takes in the Langchain api key from the .env file


app = FastAPI(
    title = "LangChain Server",
    version = "1.0",
    description = "A Simple API Server"
)


add_routes(
    app,
    Ollama(model = "gemma3:1b"),
    path = "/ollama"
)


# model = ChatOpenAI()
llm = Ollama(model = "gemma3:1b")

# prompt1 = ChatPromptTemplate.frome_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} in 100 words")

add_routes(
    app, 
    prompt2|llm,
    path  = "/poem"  ## This path is responsible in interacting with OpenAI API
    
)

if __name__ == "__main__":
    uvicorn.run(app, host = "localhost", port = 8000)

