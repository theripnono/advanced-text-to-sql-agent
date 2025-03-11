import os
import ast
import re
from typing import List
from uuid import uuid4
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.documents import Document
from config import ConfigLLM
from database import DatabaseManager
from vectorstore_manager import VectorStoreManager

llm = ConfigLLM().llm
sql_database = DatabaseManager("sqlite:///your_database.db", llm).db

# Initialize two VectorStoreManagers for different collections
vector_manager_rotulo = VectorStoreManager(db=sql_database,
                                           collection_name="rotulo_collection")
vector_manager_arrendatario = VectorStoreManager(db=sql_database,
                                                 collection_name="nombrearrendatario_collection")

"""
Retrieve SQL queries names.
If not exist, it creates two separate qdrant vectorestores to store rotulos and nombrearrendatarios
The idea is to improve "WHERE" clause avoiding incorrects or non-existing words.
Example:
    SELECT (*)
    FROM USER_NAMES
    WHERE user = "Davd" --> The name does not exist 
    
    Using the cosine vector search it can helps to find similars names,
    in that case should be: David
"""

def add_documents_to_vectorstore(documents: List[str], vector_manager):
    """Adds new documents to the specified Qdrant vector store."""
    qdrant_vectorstore = vector_manager.qdrant_vectorstore
    
    docs = [Document(page_content=doc) for doc in documents]
    uuids = [str(uuid4()) for _ in range(len(docs))]
    qdrant_vectorstore.add_documents(documents=docs, ids=uuids)

def query_as_list(query):
    """Converts SQL query results into a clean list of strings."""
    db = vector_manager_rotulo.db  # Both managers share the same db
    
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

def get_existing_documents(vector_manager):
    """Retrieves existing documents from Qdrant to avoid duplicates."""
    collection_name = vector_manager.collection_name
    existing_docs = set()
    scroll_offset = None
    while True:
        records, scroll_offset = vector_manager.qdrant_client.scroll(
            collection_name,
            limit=1000,
            offset=scroll_offset
        )
        existing_docs.update([record.payload["page_content"] for record in records if "page_content" in record.payload])
        if not scroll_offset:
            break
    return existing_docs

def setup_where_retrievers():
    
    rotulo_vectorstore = vector_manager_rotulo.qdrant_vectorstore
    existing_rotulos = get_existing_documents(vector_manager_rotulo)
    rotulos = query_as_list("SELECT rotulo FROM your_table")
    new_rotulos = [doc for doc in rotulos if doc not in existing_rotulos]
    
    if new_rotulos:
        print(f"Adding {len(new_rotulos)} new rotulos to the vector store...")
        add_documents_to_vectorstore(new_rotulos, vector_manager_rotulo)
    
    rotulo_retriever = rotulo_vectorstore.as_retriever(search_kwargs={"k": 5})
    rotulo_description = (
        "Se utiliza para buscar valores sobre los que filtrar. La entrada es una ortografía aproximada "
    )
    rotulo_tool = create_retriever_tool(
        rotulo_retriever,
        name="search_rotulos",
        description=rotulo_description,
    )

    return rotulo_tool
