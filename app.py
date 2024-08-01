import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is correctly set for the OpenAI client
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key  # Set the environment variable

def main():
    st.title("PDF Query with LangChain")
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

if __name__ == "__main__":
    main()
