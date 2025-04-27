import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import streamlit as st

from groq import Groq

def fetch_api_key():
    load_dotenv()
    return os.getenv("GROQ_API_KEY")


def extract_pdf_data(path):
    # loader = PyPDFLoader(file_path)
    loader = PyPDFLoader(path)
    doc = loader.load()
    return doc

def split_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(doc)
    return chunks
    
def create_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks,embedding)
    return vectorstore

def fetch_results(query,vectorstore):
    vectorstore_results = vectorstore.similarity_search(query,k=5)
    return vectorstore_results

def model_api_call(query,vectorstore_results):
    GROQ_API_KEY = fetch_api_key()
    client = Groq(api_key= GROQ_API_KEY)
    chatbot_instruction = "You are an expert assistant in reading context(which is PDF) and answers question from the same context. Answer the following question as clearly and concisely as possible based only on the provided context. If its not mentioned in context, answer with I don't know."
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": f"{chatbot_instruction} :{vectorstore_results}",
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": query,
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-8b-instant",

        # Optional parameters

        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=0.5,

        # The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion.
        max_completion_tokens=512,

        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,

        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,

        # If set, partial message deltas will be sent.
        stream=False,
    )
    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content