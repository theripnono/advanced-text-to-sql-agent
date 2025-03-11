import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from config import ConfigLLM
from database import DatabaseManager
import logging
from qdrant_client.http.exceptions import UnexpectedResponse
from requests.exceptions import ConnectionError

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")  # Default to local Qdrant

class VectorStoreManager:
    """
    Manages a persistent vector store using Qdrant.
    This class is responsible for:
    - Storing and retrieving embeddings.
    - Ensuring collections persist across restarts.
    """
    
    def __init__(self, db:DatabaseManager,
                 collection_name:str):
        self.db = db or None
        self.collection_name = collection_name
        self.embeddings_model = ConfigLLM().embeddings_model # text-embedding-3-large 1024
        self.qdrant_client = QdrantClient(QDRANT_HOST) # Connect to the local Qdrant Docker instance
        self._ensure_collection()
        
        self.qdrant_vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings_model,
        )

    #     # Setup retriever
    #     self.setup_retriever()

    def _ensure_collection(self):
        """Ensures the Qdrant collection exists before using it."""
        try:
            existing_collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # Same dimension as the embeddings model
                        distance=Distance.COSINE,
                    ),
                )
        except ConnectionError:
            logging.error("Failed to connect to Qdrant. Ensure the Docker container is running. 🐋")
            raise
        except UnexpectedResponse as e:
            logging.error(f"Unexpected response from Qdrant: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print("############# Ensure the Docker continer is running 🐋 ##################### ")
            raise
            
