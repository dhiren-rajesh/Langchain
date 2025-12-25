from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama

import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app=FastAPI(
    title="LangChain API",
    version="1.0.0",
    description="An API to demonstrate LangChain with Llama model",
)

llm=Ollama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("Write me a essay about {topic} with 100 words.")

add_routes(
    app,
    prompt | llm,
    path="/essay"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)