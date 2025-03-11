import os
from dotenv import load_dotenv
import getpass
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate



class ConfigLLM:
    def __init__(self):
       
        load_dotenv()
        self.setup_environment()
        self.llm=self.get_llm
        self.embeddings_model = self.get_embeddings
        
        
    def setup_environment(self):
        if not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
            os.environ["LANGSMITH_TRACING"] = "true"
            
    @property
    def get_llm(self):
        groq_api_key = os.environ["GROQ_API_KEY"]
        return ChatGroq(
            temperature=0, 
            groq_api_key=groq_api_key, 
            model_name="gemma2-9b-it"
        )
    
    @property
    def get_embeddings(self):
        
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                        dimensions=1024)  # Specify that we want 1024 embeddings (1024 dimensions). 
        return embeddings