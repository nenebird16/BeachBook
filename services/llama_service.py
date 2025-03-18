from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
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

            # Parse URI and configure for AuraDB
            uri = urlparse(NEO4J_URI)
            self.logger.debug(f"Original URI scheme: {uri.scheme}")
            self.logger.debug(f"Original URI netloc: {uri.netloc}")

            if uri.scheme == 'neo4j+s':
                url = f"bolt+s://{uri.netloc}"
                self.logger.info("Using AuraDB connection format")
            else:
                url = NEO4J_URI
                self.logger.info("Using standard Neo4j connection format")

            self.logger.debug(f"Final connection URI (without credentials): {url}")

            # Define Cypher query templates
            cypher_queries = {
                "node_context": "MATCH (n:Document) WHERE n.content CONTAINS $keyword RETURN n.content AS content",
                "rel_context": """
                    MATCH (d:Document)-[r]-(e:Entity)
                    WHERE d.content CONTAINS $keyword
                    RETURN d.content AS content, type(r) as relationship, e.name as entity
                """,
                "keyword_context": """
                    MATCH (d:Document)
                    WHERE any(keyword IN $keywords WHERE d.content CONTAINS keyword)
                    RETURN d.content AS content
                """
            }

            # Initialize graph store with query templates
            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=url,
                database="neo4j",
                cypher_queries=cypher_queries
            )

            self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            self.logger.info("Successfully initialized Neo4j graph store with query templates")

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j graph store: {str(e)}")
            raise

    def process_document(self, content):
        """Process document content and create index"""
        try:
            self.logger.info("Processing document with LlamaIndex")
            documents = [Document(text=content)]
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context
            )
            self.logger.info("Successfully processed document")
            return self.index
        except Exception as e:
            self.logger.error(f"Error processing document with LlamaIndex: {str(e)}")
            raise

    def process_query(self, query_text):
        """Process a query using the RAG pipeline"""
        try:
            self.logger.info(f"Processing query: {query_text}")

            # Create retriever with Neo4j-aware configuration
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3,
                # Include graph-based filters
                filters={
                    "node_types": ["Document", "Entity"],
                    "edge_types": ["CONTAINS"]
                }
            )

            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever
            )

            # Execute query
            response = query_engine.query(query_text)

            # Log response metadata
            self.logger.info("Query processed successfully")
            self.logger.debug(f"Response source nodes: {len(response.source_nodes)}")

            # Format response with sources and relevant relationships
            formatted_response = str(response)
            if hasattr(response, 'source_nodes') and response.source_nodes:
                formatted_response += "\n\nSources and Related Entities:\n"
                for idx, node in enumerate(response.source_nodes, 1):
                    formatted_response += f"{idx}. Content: {node.node.text[:200]}...\n"
                    # Include related entities if available
                    if hasattr(node, 'relationships'):
                        formatted_response += "   Related entities: "
                        formatted_response += ", ".join([rel['entity'] for rel in node.relationships])
                        formatted_response += "\n"

            return formatted_response

        except AttributeError as e:
            if "'LlamaService' object has no attribute 'index'" in str(e):
                return "Please upload a document first before querying."
            self.logger.error(f"Error processing query: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise