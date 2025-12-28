import streamlit as st
import os
from dotenv import load_dotenv
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.final_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(st.session_state.docs[:50])
    st.session_state.vector_docs = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

st.title("Langchain with Groq LLM and Ollama Embeddings")

groq_llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the context provided.
    Please provide detailed and informative answers.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm=groq_llm, prompt=prompt)
retriever = st.session_state.vector_docs.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Enter your question about Langchain:")

if prompt:
    start=time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response Time:", time.process_time() - start)
    st.write("Response:", response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


