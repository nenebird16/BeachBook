from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
import logging
from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from urllib.parse import urlparse

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = OPENAI_API_KEY

        # Initialize Neo4j graph store
        try:
            # Parse the URI
            uri = urlparse(NEO4J_URI)
            if uri.scheme == 'neo4j+s':
                # For AuraDB, use bolt+s://
                netloc = uri.netloc
                bolt_uri = f"bolt+s://{netloc}"
                self.logger.info(f"Using AuraDB connection with URI: {bolt_uri}")
            else:
                bolt_uri = NEO4J_URI
                self.logger.info(f"Using standard connection with URI: {bolt_uri}")

            self.logger.debug(f"Attempting to connect to Neo4j with user: {NEO4J_USERNAME}")
            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                url=bolt_uri,
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