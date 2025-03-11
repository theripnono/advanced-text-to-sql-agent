import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DatabaseManager:
    def __init__(self, db_url: str, llm):
        
        self.engine = create_engine(db_url)
        self.db = SQLDatabase(self.engine)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
        self.tools = self.toolkit.get_tools()
    
    def load_columns_context(self):
        with open("columns_explanation.txt", "r", encoding="utf-8") as file:
            return file.read()
