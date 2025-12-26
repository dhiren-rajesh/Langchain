import requests
import streamlit as st

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={"input": {"topic": input_text}})
    
    return response.json()["output"]

st.title("LangChain API Client with Ollama")
input_text = st.text_input("Enter a topic to generate an essay about")

st.write(get_ollama_response(input_text))