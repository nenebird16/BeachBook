from llama_index.core import Settings
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

            # Parse URI for AuraDB
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

            # Define Cypher query templates for graph queries
            cypher_queries = {
                "content_query": """
                    MATCH (d:Document)
                    WHERE d.content CONTAINS $query
                    WITH d, score() as relevance
                    MATCH (d)-[r:CONTAINS]->(e:Entity)
                    RETURN d.content as content, 
                           d.title as title,
                           collect(distinct e.name) as entities,
                           relevance
                    ORDER BY relevance DESC
                    LIMIT 5
                """,
                "entity_query": """
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query
                    WITH e
                    MATCH (d:Document)-[:CONTAINS]->(e)
                    RETURN d.content as content,
                           d.title as title,
                           collect(distinct e.name) as entities
                    LIMIT 5
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
            self.logger.info("Successfully initialized Neo4j graph store")

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j graph store: {str(e)}")
            raise

    def process_document(self, content):
        """Process document content for storage"""
        try:
            self.logger.info("Processing document for storage")
            # Note: Document processing now handled by DocumentProcessor
            return True
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def process_query(self, query_text):
        """Process a query using the graph knowledge base"""
        try:
            self.logger.info(f"Processing query: {query_text}")

            # Try content-based search first
            content_results = self.graph_store.raw_query(
                query_text=query_text,
                query_type="content_query",
                parameters={"query": query_text}
            )
            self.logger.debug(f"Content query results: {content_results}")

            # Try entity-based search
            entity_results = self.graph_store.raw_query(
                query_text=query_text,
                query_type="entity_query",
                parameters={"query": query_text}
            )
            self.logger.debug(f"Entity query results: {entity_results}")

            # Combine and format results
            response = "Here's what I found in the knowledge graph:\n\n"

            if content_results:
                response += "Content matches:\n"
                for idx, result in enumerate(content_results, 1):
                    response += f"{idx}. Document: {result['title']}\n"
                    response += f"   Content: {result['content'][:200]}...\n"
                    if result['entities']:
                        response += f"   Related concepts: {', '.join(result['entities'])}\n"
                    response += "\n"

            if entity_results:
                response += "\nEntity matches:\n"
                for idx, result in enumerate(entity_results, 1):
                    response += f"{idx}. Found in document: {result['title']}\n"
                    response += f"   Context: {result['content'][:200]}...\n"
                    response += f"   Related concepts: {', '.join(result['entities'])}\n"
                    response += "\n"

            if not content_results and not entity_results:
                response = "I couldn't find any relevant information in the knowledge graph for your query."

            return response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise