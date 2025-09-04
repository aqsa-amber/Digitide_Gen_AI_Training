import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings # <--- This is the new import

# --- 1. Get Groq API Key from Streamlit Secrets ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- 2. Prepare Data and Vector Store (No OpenAI Key Needed) ---
@st.cache_resource
def setup_vector_store():
    """Loads documents and creates a vector store."""
    data_files = ['salary.txt', 'insurance.txt']
    documents = []
    for file in data_files:
        if os.path.exists(file):
            loader = TextLoader(file)
            documents.extend(loader.load())
        else:
            st.error(f"Required file not found: {file}")
            st.stop()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Use HuggingFaceEmbeddings to run a local model
    # This does not require any API key
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb

vectordb = setup_vector_store()

# --- 3. Define Agents ---
@st.cache_resource
def setup_agents(llm_model):
    """Sets up the retrieval agents with the language model."""
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)

    salary_agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )

    insurance_agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    return salary_agent, insurance_agent

# Use the 'mixtral-8x7b-32768' model for fast and high-quality responses
salary_agent, insurance_agent = setup_agents("llama3-8b-8192")

# --- 4. Coordinator Agent Logic ---
def coordinator_agent(query):
    """Routes the query to the correct agent based on keywords."""
    query_lower = query.lower()
    if "salary" in query_lower or "deduction" in query_lower or "income" in query_lower:
        st.info("Routing query to Salary Agent...")
        return salary_agent.run(query)
    elif "insurance" in query_lower or "policy" in query_lower or "coverage" in query_lower:
        st.info("Routing query to Insurance Agent...")
        return insurance_agent.run(query)
    else:
        return "I'm sorry, I can only answer questions about salary or insurance."

# --- 5. Streamlit App UI ---
st.title("Multi-Agent RAG System with Groq (Local Embeddings)")
st.write("Ask a question about your salary or insurance.")

query_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if query_input:
        with st.spinner("Processing..."):
            try:
                response = coordinator_agent(query_input)
                st.success("Here's your answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")

