# split_docs.py
import sys
from pathlib import Path

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(path: Path):
    if path.suffix.lower() == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    elif path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    else:
        raise ValueError("Unsupported file type. Use a .txt or .pdf file.")
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    return chunks

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_docs.py /path/to/file.pdf (or .txt)")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    docs = load_documents(path)
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

    print(f"Loaded {len(docs)} document(s).")
    print(f"Total chunks created: {len(chunks)}")

if __name__ == "__main__":
    main()
