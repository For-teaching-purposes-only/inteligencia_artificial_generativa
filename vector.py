from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


df = pd.read_csv("realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "/chrome_langchain_db"
add_documents = not os.path.exists(db_location)

#Title,Date,Rating,Review
if add_documents:
    documents = []
    ids = []
    print('entra aqui alguma vez')
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"],
            metadata={"Review": row["Review"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="realistic_restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)   

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k":5}
)
