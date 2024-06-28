import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_ollama_embedding_function
from langchain_community.vectorstores import Chroma

import time

CHROMA_PATH = "./chroma"
DATA_PATH = "./data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # start timer to measure time taken to populate the database
    start = time.time()

    # Create (or update) the data store.
    documents = load_documents()
    print(f"Number of documents to add: {len(documents)}")
    temp1 = time.time()
    print(f"Time taken to load documents: {temp1 - start} seconds")

    chunks = split_documents(documents)
    print(f"Number of chunks to add: {len(chunks)}")
    temp2 = time.time()
    print(f"Time taken to split documents: {temp2 - temp1} seconds")

    add_to_chroma(chunks)
    print("Database updated")

    # end timer
    end = time.time()
    print(f"Time taken to load/split documents and add to database: {end - start} seconds")


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    print(f"Loading documents from {DATA_PATH}")

    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    time1 = time.time()
    print(f"start time: {time1} seconds, before loading the existing database")
    # Load the existing database.
    db = Chroma(
        # persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        persist_directory=CHROMA_PATH, embedding_function=get_ollama_embedding_function()
    )
    time2 = time.time()
    print(f"Time taken to load the existing database: {time2 - time1} seconds")

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    time3 = time.time()
    print(f"Time taken to calculate chunk IDs: {time3 - time2} seconds")

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    # print(f"Number of existing documents in DB: {len(existing_ids)}")

    time4 = time.time()
    print(f"Time taken to get existing documents from database: {time4 - time3} seconds")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    time5 = time.time()
    print(f"Time taken to compare existing documents with new documents: {time5 - time4} seconds")

    if len(new_chunks):
        print(f"New documents being added:")
        print(f"Adding new documents: {len(new_chunks)} chunks")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        start = time.time()
        # print(f"start time: {start} seconds, before adding new documents to database")

        db.add_documents(new_chunks, ids=new_chunk_ids)

        temp = time.time()
        print(f"Time taken to add new documents to database: {temp - start} seconds")

        # db.persist()
        end = time.time()
        print(f"Time taken to persist the database: {end - temp} seconds")
        # print("New documents added")
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
    # clear_database()
