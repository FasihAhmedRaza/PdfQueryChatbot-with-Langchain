import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import os
import openai

# Set the password and the OpenAI API key in the backend
backend_password = "10"
backend_openai_api_key = "sk-proj-SD4O6s1SWoXLnmuZe6LtT3BlbkFJEneWZSk8hLv3h1oXCfjK"

def validate_openai_key(api_key):
    try:
        openai.api_key = api_key
        # Perform a simple test API call
        openai.Completion.create(
            engine="text-davinci-003",
            prompt="This is a test prompt.",
            max_tokens=5
        )
        return True
    except openai.error.AuthenticationError:
        st.error("Invalid API key.")
        return False
    except Exception as e:
        st.error(f"API Key validation error: {e}")
        return False

def main():
    st.title("PDF Query with LangChain")

    # Prompt the user for a password
    entered_password = st.text_input("Enter the password to use Chatbot:", type="password")

    if entered_password == backend_password:
        st.success("Password correct! .")
        openai.api_key = backend_openai_api_key

        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file:
            pdfreader = PdfReader(uploaded_file)
            raw_text = ""
            for i, page in enumerate(pdfreader.pages):
                content = page.extract_text()
                if content:
                    raw_text += content

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)

            embeddings = OpenAIEmbeddings()
            document_search = FAISS.from_texts(texts, embeddings)

            st.write("PDF processed successfully!")

            query = st.text_input("Enter your query:")
            if query:
                chain = load_qa_chain(OpenAI(), chain_type="stuff")
                docs = document_search.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)
                st.write(response)
    else:
        st.warning("Incorrect password. Please try again.")

if __name__ == "__main__":
    main()
