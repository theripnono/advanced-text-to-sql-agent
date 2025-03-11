import os
import re
from uuid import uuid4
from typing_extensions import Annotated
from typing import TypedDict, List


from config import ConfigLLM
from vectorstore_manager import VectorStoreManager
from langchain.schema import Document
# from langchain.agents.agent_toolkits import create_retriever_tool

# Initialize the VectorStoreManager
vector_manager = VectorStoreManager(db=None, collection_name="sql_examples")

# File to store the last modification timestamp
TIMESTAMP_FILE = "last_modified.txt"

class SqlMetadata(TypedDict):
    number: Annotated[int, "Número de la SQL"]
    title: Annotated[str, "Título de la SQL"]
    description: Annotated[str, "Corta descripción de la query"]

def generate_sql_metadata(sql_query: str):
    prompt = (
        "Dada la siguiente SQL, tu deber es hallar su metadata en formato estructurado:\n"
        "Devuelve el resultado en JSON estricto asegurando que el número sea un entero.\n"
        "El metadata está compuesto por:\n"
        "-number (int): Un número identificador de la query.\n"
        "-title (str): El título de la query.\n"
        "-description (str): Una breve explicación de lo que hace la query.\n"
        f"SQL:\n{sql_query}\n"
    )
    structured_llm = ConfigLLM().llm.with_structured_output(SqlMetadata)
    
    try:
        result = structured_llm.invoke(prompt)
        result["number"] = int(result.get("number", 0))
        return result
    except Exception as e:
        print(f"Error processing SQL metadata: {e}")
        return {"number": 0, "title": "Error", "description": "Failed to process metadata"}

def extract_sql_query(text: str) -> str:
    """Extracts only the SQL query (starting from SELECT) from a given text."""
    match = re.search(r"(?i)\bSELECT\b.*", text, re.DOTALL)
    return match.group(0).strip() if match else ""

def extract_sql_title(text: str) -> str:
    """
    Extracts only the title part (before SELECT) from a given text.
    """
    match = re.search(r"(?i)^(.*?)(\bSELECT\b)", text, re.DOTALL)
    if match:
        title = match.group(1).strip()
        return title
    return ""

def enrich_sql_metadata(query: str, basic_metadata: SqlMetadata) -> dict:
    """Adds additional metadata by analyzing the SQL query."""
    tables = re.findall(r'(?i)FROM\s+([a-z0-9_]+)', query)
    tables.extend(re.findall(r'(?i)JOIN\s+([a-z0-9_]+)', query))
    
    has_aggregation = bool(re.search(r'(?i)(COUNT|SUM|AVG|MIN|MAX)\s*\(', query))
    has_groupby = bool(re.search(r'(?i)GROUP\s+BY', query))
    has_orderby = bool(re.search(r'(?i)ORDER\s+BY', query))
    
    return {
        **basic_metadata,
        "tables_referenced": list(set(tables)),
        "query_type": "SELECT",
        "has_aggregation": has_aggregation,
        "has_groupby": has_groupby,
        "has_orderby": has_orderby,
        "complexity": calculate_complexity(query),
        "query": query
    }

def calculate_complexity(query: str) -> str:
    """Estimates query complexity based on features."""
    score = 0
    if "JOIN" in query.upper(): score += 2
    if "GROUP BY" in query.upper(): score += 1
    if "HAVING" in query.upper(): score += 1
    if "UNION" in query.upper(): score += 2
    if "SUBQUERY" in query.upper() or "SELECT" in query.upper().split("FROM")[0]: score += 3
    
    if score <= 2: return "simple"
    elif score <= 5: return "moderate"
    else: return "complex"

def parse_sql_file(file_path) -> list[Document]:
    """Parses a SQL file and returns a list of Document objects."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    sql_statements = content.split(';')
    documents = []
    for sql in sql_statements:
        
        sql = sql.strip()
        query = extract_sql_query(sql)
        title = extract_sql_title(sql)
        
        print(f"Parsing {title}...")
        
        if not query:
            continue
        
        metadata = generate_sql_metadata(sql)
        enriched_metadata = enrich_sql_metadata(query, metadata)
        
        doc = Document(
            page_content=title,
            metadata=enriched_metadata
        )
        documents.append(doc)
    
    return documents

def get_last_modified_time(file_path: str) -> float:
    """Returns the last modification time of a file."""
    return os.path.getmtime(file_path)

def save_last_modified_time(file_path: str, timestamp: float):
    """Saves the last modification time of a file to a timestamp file."""
    with open(TIMESTAMP_FILE, 'w') as f:
        f.write(str(timestamp))

def load_last_modified_time() -> float:
    """Loads the last modification time from the timestamp file."""
    if not os.path.exists(TIMESTAMP_FILE):
        return 0
    with open(TIMESTAMP_FILE, 'r') as f:
        return float(f.read().strip())

def is_file_modified(file_path: str) -> bool:
    """Checks if the file has been modified since the last recorded timestamp."""
    last_modified = get_last_modified_time(file_path)
    saved_modified = load_last_modified_time()
    return last_modified > saved_modified

def add_documents_to_vectorstore(documents: List[Document]):
    """Adds new documents to the Qdrant vector store."""
    qdrant_vectorstore = vector_manager.qdrant_vectorstore
    existing_docs = get_existing_documents(vector_manager.collection_name)
    new_docs = [doc for doc in documents if doc.page_content not in existing_docs]
    
    if new_docs:
        print(f"Adding {len(new_docs)} new documents to the vector store...")
        uuids = [str(uuid4()) for _ in range(len(new_docs))]
        qdrant_vectorstore.add_documents(documents=new_docs, ids=uuids)
    else:
        print("No new documents to add.")

def get_existing_documents(collection_name):
    """Retrieves existing documents from Qdrant to avoid duplicates."""
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

def setup_examples_retriever(path)->object:
    """Sets up the retriever tool, processing the SQL file only if it has been modified."""
    if not is_file_modified(path):
        print("SQL file has not been modified. Using existing vector store.")
        
    else:
        print("SQL file has been modified. Reprocessing...")
        parsed_docs = parse_sql_file(path)
        add_documents_to_vectorstore(parsed_docs)
        save_last_modified_time(path, get_last_modified_time(path))
    
    qdrant_vectorstore = vector_manager.qdrant_vectorstore
    retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 3},)
    
    return retriever
