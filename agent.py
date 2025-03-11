import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing_extensions import Annotated
from typing import TypedDict

from langchain import hub
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate

from config import ConfigLLM
from database import DatabaseManager

from sqlexamples_retriver import setup_examples_retriever
from sqlwhere_retriver import setup_where_retrievers

from datetime import datetime 



class DatabaseMetadataHelper :
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.get_columns_context = db.load_columns_context()
        self.table_description = self.get_table_info()
        
        
    def get_table_info(self):
        with open("table_description.txt", "r") as f:
            table_description =f.read()
        return table_description
    
class RetrievalEngine:

    def __init__(self):
        self.sql_examples_retriever = setup_examples_retriever("sql_examples.txt")
        self.where_filter_retriever = setup_where_retrievers()

llm = ConfigLLM().llm
dm = DatabaseManager("sqlite:///rent_roll.db", llm)
load_columns_context = dm.load_columns_context()

db_helper = DatabaseMetadataHelper (dm, llm)

retriever_engine = RetrievalEngine()
class AgentState(TypedDict):
    question: str
    query: str
    dangerous:str
    query_error: bool
    attempts: int
    answer: str
    aboutme: str
    result:str
    prevent_answer:str
    rewrittenquestion:str
    normalized_where:str
    language:str # TODO: add lang in metadata
    
class QueryOutput(TypedDict):
    query: Annotated[str, "Consulta SQL sintácticamente válida.."]

class ValidationOutput(TypedDict):
    dangerous: Annotated[str, "Indica si la pregunta es maliciosa o peligros parala base de datos. Devuelve 'si' o 'no'."]
        
class RewrittenQuestion(TypedDict):
    rewrittenquestion:Annotated[str, "Reescribir pregunta"]

class WhoIAmAnswer(TypedDict):
    answer:Annotated[str, "Presentación del agente"]

class PreventDangerousQuestion(TypedDict):
    prevent_answer:Annotated[str, "Respuesta a pregunta peligrosa"]

class NormalizedWhereClause(TypedDict):
    normalized_where:Annotated[str, "Nombre propio que contiene la base de datos"]

def create_logs(**kwargs):
    """
    Create logs with the given keyword arguments.

    Args:
    **kwargs: The keyword arguments to log.

    Returns:
    None
    """
    timestamp = datetime.now()
    with open('logs.txt', 'a', encoding='utf-8') as file:
        file.write(f"{timestamp}\n")
        for key, value in kwargs.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")

def check_user_query(state:AgentState):
 
    question = state["question"]
    
    print(f"Analizando la pregunta... << {question} >>")
    
    check_user_query_prompt_template = hub.pull("check-user-question")
    prompt = check_user_query_prompt_template.invoke(
        {
            "question":question
        }
    )
    
    structured_llm = llm.with_structured_output(ValidationOutput)
    result = structured_llm.invoke(prompt)
    
    return {"dangerous": result["dangerous"]}

def prevent_dangerous_query(state:AgentState):
    
    question = state["question"]
    prevent_dangerous_question_prompt_template = hub.pull("prevent-dangerous-question")
    
    prompt = prevent_dangerous_question_prompt_template.invoke(
        {
            "question":question
        }
    )
    
    structured_llm = llm.with_structured_output(PreventDangerousQuestion)
    result = structured_llm.invoke(prompt)
    
    return {"prevent_answer": result["prevent_answer"]}

def get_similar_queries(question):
    "This function use RAG and it retrieves similar input queries."
    
    retriever = retriever_engine.sql_examples_retriever
    documents = retriever.invoke(question)

    similar_queries_context_list = [
        {"titulo": doc.page_content, "metadata": doc.metadata} for doc in documents
    ]
    
    return similar_queries_context_list

def prune_ddl_query(state:AgentState):
    """
    This function avoids extended ddl in order to request less tokens.
    It returns necessary columns.
    """
    # TODO: Use this funtion with more tables
    question = state["question"]
    
    prompt = (
        f"Dada la pregunta: question:{question}, el ddl con las explicaciones de la tabla, y la explicación  de la columna,"
        "Devuelve unicamente la información necesaria del dll y las columas que necesites para la query" 
            f' "table_ddl_info": {dm.db.get_table_info()}\n'
            f' "table_explanation": {load_columns_context}\n'

        )
    response = llm.invoke(prompt)
    
    return response

def collention_name_helper(question):
    "Determine if need to filter"
    
    query_prompt_template = PromptTemplate.from_template(
       "Dada la consulta: {question}, determina si el filtro debe aplicarse'.\n"
        "Tu respuesta debe ser exclusivamente el nombre: sin explicaciones adicionales."
    )
    
    prompt = query_prompt_template.invoke({
        "question":question,
    })
    
    structured_llm = llm.with_structured_output(NormalizedWhereClause)
    
    collention = structured_llm.invoke(prompt)
    return {"normalized_where":collention['normalized_where']}
    
def normalize_where_clause(question):
    """
    Use retriever to improve the "WHERE" clause.
    This function corrects user input mistakes, such as typos and capitalization errors,
    ensuring the correct name is used in SQL queries.

    Example:
        - User input: "Davd" (typo) → Corrected: "David"
        - User input: "david" (wrong capitalization) → Corrected: "David"

    This prevents SQL errors when filtering by name.
    """

    name_from_collection= collention_name_helper(question)

    retriever = retriever_engine.where_filter_retriever(name_from_collection)
    retrieved = retriever.invoke(question).split("\n\n")

    query_prompt_template = PromptTemplate.from_template(
        "Eres un experto en SQL. Tu tarea es encontrar el nombre correcto para una cláusula WHERE. "
        "El usuario podría haber escrito mal el nombre o usar una capitalización incorrecta.\n\n"
        "### Datos de referencia:\n"
        "- Pregunta original: {question}\n"
        "- Lista de nombres disponibles : {retrieved}\n\n"
        "### Instrucciones:\n"
        "1. Encuentra el nombre más parecido dentro de la lista proporcionada.\n"
        "2. Devuelve solo el nombre correcto.\n"
        "Devuelve unicamente el nombre"
    )

    prompt = query_prompt_template.invoke({
        "question": question,
        "retrieved": retrieved
    })
    
    structured_llm = llm.with_structured_output(NormalizedWhereClause)
  
    corrected_name = structured_llm.invoke(prompt)
    
    print(f"Sanitazing where clausule: {corrected_name['normalized_where']}")

    return {"normalized_where":corrected_name['normalized_where']}

def write_query(state: AgentState):

    question = state["question"]

    similar_queries = get_similar_queries(question)

    query_prompt_template = hub.pull("sql-query-system-prompt-es")
    
    prompt = query_prompt_template.invoke(
        {
            "dialect": dm.db.dialect,
            "top_k": 5,
            "table_info": dm.db.get_table_info(), # get DDL + table 3 rows + columns
            "table_explanation": load_columns_context,
            "table_description": db_helper.table_description,
            "similar_queries": similar_queries,
            "where_clause": normalize_where_clause(question),
            "input": question,
        }
    )

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)

    return {"query": result["query"]}

def execute_query(state: AgentState):
    
    sql_query = state["question"].strip()

    print(f"Executing SQL query: {sql_query} ---> {state["query"]}")
    
    execute_query_tool = QuerySQLDatabaseTool(db=dm.db)
    
    executed_query= execute_query_tool.invoke(state["query"])
    
    
    if executed_query.startswith("Error"):
        return {"result" : executed_query,
                "query_error" : True}
    else:
        return {"result": executed_query,
                "query_error" : False}

def generate_answer(state: AgentState):
    
    if state["query_error"]:
        prompt = (
            "Lo siento, no se ha podidp ejecutar la consulta"
            f'Consulta {state["query"]}\n'
            f'Error: {state["result"]}\n'
        )
        response = llm.invoke(prompt)

        return {"answer": response.content}

    prompt = (
        "Dada la siguiente pregunta de usuario, la correspondiente consulta SQL, "
        "y el resultado SQL, responde a la pregunta del usuario.\n\n"
        f'Pregunta: {state["question"]}\n'
        f'Consulta SQL: {state["query"]}\n'
        f'Resultado SQL: {state["result"]}'
    )
    response = llm.invoke(prompt)
    
    create_logs(
        question=state["question"],
        query=state["query"],
        result=state["result"],
        answer=response.content
    )
    
    return {"answer": response.content}

def regenerate_query(state:AgentState):
    
    if "attempts" not in state:
        state["attempts"] = 0
    
    question = state["question"]
    
    print(f"Reescribiendo la pregunta... << {question} >>")
    
    rewritten_question_prompt_template = hub.pull("rewritten-user-question")
    
    prompt = rewritten_question_prompt_template.invoke(
        {
            "question": question,
            "query": state["query"],
            "result": state["result"],
        }
    )
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewritten = structured_llm.invoke(prompt)
        
    state["question"] = rewritten["rewrittenquestion"]
    state["attempts"] += 1
    
    print(f"Rewritten question: {state['question']}")
    
    return {"answer": rewritten.get("rewrittenquestion"),
            "attempts": state["attempts"]}

def end_max_iterations(state: AgentState):
    
    state["answer"] = "Please try again."
    print("Maximum attempts reached. Ending the workflow.")
    
    return {"answer": state["answer"]}

def relevance_router(state: AgentState):
    if state["dangerous"].lower() == "no":
        return "write_query"
    else:
        return "prevent_dangerous_query"

def check_attempts_router(state: AgentState):
    if state["attempts"] < 3:
        return "write_query"
    else:
        return "max_iterations"

def execute_sql_router(state: AgentState):

    if not state.get("query_error", False): # Get method return False by default
        return "execute_query"
    else:
        return "regenerate_query"


workflow = StateGraph(AgentState)

workflow.add_node("write_query", write_query)
workflow.add_node("check_user_query", check_user_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)

workflow.add_node("prevent_dangerous_query", prevent_dangerous_query)

workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("end_max_iterations", end_max_iterations)

workflow.add_node("check_attempts_router", check_attempts_router)
workflow.add_node("execute_sql_router", execute_sql_router)


workflow.add_edge(START,"check_user_query")


workflow.add_conditional_edges(
    "check_user_query",
    relevance_router,
    {
        "write_query": "write_query",
        "prevent_dangerous_query": "prevent_dangerous_query",
    },
)

workflow.add_edge("write_query", "execute_query")

workflow.add_conditional_edges(
    "execute_query",
    execute_sql_router,
    {
        "execute_query": "generate_answer",
        "regenerate_query": "regenerate_query",
    },
)

workflow.add_conditional_edges(
    "regenerate_query",
    check_attempts_router,
    {
        "write_query": "write_query",
        "max_iterations": "end_max_iterations",
    },
)

workflow.add_edge("prevent_dangerous_query", END)
workflow.add_edge("generate_answer", END)
workflow.add_edge("end_max_iterations", END)

memory = MemorySaver()
config = {
    "configurable":
        {"thread_id": "testv2-thread"
         }
    }



graph = workflow.compile() # checkpointer=memory, interrupt_before=["write_query"] 


# TODO: Improve persistence in memory for the agent


question = "Your question...."

for step in graph.stream(
    {"question": question},
    config=config,
    stream_mode="updates"
):
    print(step)