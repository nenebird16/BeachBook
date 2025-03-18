from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
import logging
from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = OPENAI_API_KEY

        # Initialize Neo4j graph store
        try:
            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=NEO4J_URI,
                database="neo4j"
            )
            self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            self.logger.info("Successfully initialized Neo4j graph store")
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j graph store: {str(e)}")
            raise

    def process_document(self, content):
        """Process document content and create index"""
        try:
            self.logger.info("Processing document with LlamaIndex")
            documents = [Document(text=content)]
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context
            )
            self.logger.info("Successfully processed document")
            return index
        except Exception as e:
            self.logger.error(f"Error processing document with LlamaIndex: {str(e)}")
            raise

    def process_query(self, query_text):
        """Process a query using the RAG pipeline"""
        try:
            self.logger.info(f"Processing query: {query_text}")
            # Create query engine from graph store
            query_engine = self.graph_store.as_query_engine()
            # Execute query
            response = query_engine.query(query_text)
            self.logger.info("Successfully processed query")
            return str(response)
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise