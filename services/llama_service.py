from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
import logging
from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from urllib.parse import urlparse

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = OPENAI_API_KEY

        # Initialize Neo4j graph store
        try:
            if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                raise ValueError("Neo4j credentials not properly configured")

            # Parse the URI and extract hostname for AuraDB
            uri = urlparse(NEO4J_URI)
            self.logger.debug(f"Original URI scheme: {uri.scheme}")
            self.logger.debug(f"Original URI netloc: {uri.netloc}")

            # Configure connection for AuraDB
            if uri.scheme == 'neo4j+s':
                url = f"bolt+s://{uri.netloc}"
                self.logger.info("Using AuraDB connection format")
            else:
                url = NEO4J_URI
                self.logger.info("Using standard Neo4j connection format")

            self.logger.debug(f"Final connection URI (without credentials): {url}")
            self.logger.debug(f"Attempting to connect with user: {NEO4J_USER}")

            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=url,
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

            # Create query engine with more specific parameters
            query_engine = self.graph_store.as_query_engine(
                response_mode="tree_summarize",
                verbose=True,
                similarity_top_k=3  # Retrieve top 3 most relevant nodes
            )

            # Execute query
            response = query_engine.query(query_text)

            # Log response metadata
            self.logger.info("Query processed successfully")
            self.logger.debug(f"Response source nodes: {len(response.source_nodes)}")

            # Format response with sources
            formatted_response = str(response)
            if hasattr(response, 'source_nodes') and response.source_nodes:
                formatted_response += "\n\nSources:\n"
                for idx, node in enumerate(response.source_nodes, 1):
                    formatted_response += f"{idx}. {node.node.text[:200]}...\n"

            return formatted_response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise