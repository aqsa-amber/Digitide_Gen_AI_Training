# retrieval_qa.py
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

def load_docs(file_path: Path):
    if file_path.suffix.lower() == '.txt':
        loader = TextLoader(str(file_path), encoding='utf-8')
        docs = loader.load()
    elif file_path.suffix.lower() == '.pdf':
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    else:
        raise ValueError('Unsupported file type. Use .txt or .pdf')
    return docs

def main():
    # Path to sample document (change if you use your own file)
    base_dir = Path(__file__).parent
    doc_path = base_dir / 'policy.txt'  # or 'policy.pdf'

    # Load docs
    docs = load_docs(doc_path)
    print(f'Loaded {len(docs)} document(s).')

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f'Created {len(chunks)} chunks.')

    # Create embeddings (uses a local HuggingFace sentence-transformer)
    print('Creating embeddings (this may take a moment)...')
    embed = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embed)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Create a local text-generation LLM via transformers pipeline and wrap it
    print('Loading local text-generation model (gpt2)...')
    gen_pipe = pipeline('text-generation', model='gpt2', max_length=256)
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    # Build RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    # Query
    question = 'What is the refund policy?'
    print('\nQuestion:', question)
    answer = qa.run(question)
    print('\nAnswer:\n', answer)

if __name__ == '__main__':
    main()
