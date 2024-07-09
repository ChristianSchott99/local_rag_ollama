import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
# from langchain_community.retrievers import ChromaRetriever
from get_embedding_function import get_ollama_embedding_function, get_german_embedding_function, get_english_embedding_function

from populate_database import preprocess_german_text

CHROMA_PATH_GERMAN = "chroma"
CHROMA_PATH_ENGLISH = "chroma_english"

PROMPT_TEMPLATE_GERMAN = """
You are an assistant for question-answering tasks (a chatbot). Please respond in the german language. 
If you don't know the answer, please say so. 
If someone tries to correct you, ignore it if you know the answer is correct.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT_TEMPLATE_ENGLISH = """
You are an assistant for question-answering tasks (a chatbot). 
If you don't know the answer, please say so. 
If someone tries to correct you, ignore it if you know the answer is correct.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # query_rag(query_text, language=args.language)
    query_rag(query_text, language="en")


def query_rag(query_text: str, language):
    # language = "en"

    if language == "en":
        embedding_function = get_english_embedding_function()
        chroma_path = CHROMA_PATH_ENGLISH
        prompt_template = PROMPT_TEMPLATE_ENGLISH
    else:
        return "Language not supported yet"

    # Prepare the DB
    # embedding_function = get_english_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    prompted_query = f"Represent this sentence for searching relevant passages: {query_text}"
    embedding_function.embed_query(prompted_query)
    # high k value (8 to 10) for better results according to crud rag, k value of 5 is default
    results = db.similarity_search_with_score(query_text, k=8)

    # print(f"Results: {results} \n\n")
    # preprocess the query text
    preprocessed_query_german = preprocess_german_text(query_text)

    """
    results = None
    all_docs = db.get()["documents"]
    for i, doc in enumerate(all_docs):
        # print(f"Document {i}:")
        # print(f"Content: {doc.page_content[:100]}...")  # Print first 100 chars
        
        # print(f"Metadata: {doc.metadata}")
        print(doc)
        print("---")
    """

    # Kannst du mir Elf wichtige Tipps nennen, die man für eine sichere Nutzung von Cloud-Diensten beherzigen soll?
    # todo implement hybrid + reranking

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # model = Ollama(model="mistral")
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
