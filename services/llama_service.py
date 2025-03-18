from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jGraphStore
import logging
from config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from urllib.parse import urlparse
from py2neo import Graph, ConnectionProfile

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = OPENAI_API_KEY

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

            # Initialize direct Neo4j connection for queries
            profile = ConnectionProfile(
                scheme="bolt+s" if uri.scheme == 'neo4j+s' else uri.scheme,
                host=uri.netloc,
                port=7687,
                secure=True if uri.scheme == 'neo4j+s' else False,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            self.graph = Graph(profile=profile)
            self.logger.info("Successfully connected to Neo4j database")

            # Initialize LlamaIndex graph store for document processing
            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=url,
                database="neo4j"
            )
            self.logger.info("Successfully initialized Neo4j graph store")

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j connections: {str(e)}")
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
        """Process a query using semantic understanding and graph knowledge base"""
        try:
            self.logger.info(f"Processing query: {query_text}")
            
            # Semantic query understanding
            query_embedding = self.embed_model.get_text_embedding(
                text=query_text
            )
            
            # Hybrid search combining semantic and graph
            semantic_query = """
            CALL db.index.vector.queryNodes('document_embeddings', 10, $embedding)
            YIELD node, score
            WITH node, score
            MATCH (node)-[:CONTAINS]->(e:Entity)
            RETURN node.content as content, 
                   node.title as title,
                   collect(distinct e.name) as entities,
                   score as relevance
            ORDER BY score DESC
            LIMIT 5
            """

            # Content-based search
            content_query = """
                MATCH (d:Document)
                WHERE toLower(d.content) CONTAINS toLower($query)
                MATCH (d)-[r:CONTAINS]->(e:Entity)
                RETURN d.content as content, 
                       d.title as title,
                       collect(distinct e.name) as entities,
                       count(e) as relevance
                ORDER BY relevance DESC
                LIMIT 5
            """
            content_results = self.graph.run(content_query, query=query_text).data()
            self.logger.debug(f"Content query results: {content_results}")

            # Entity-based search
            entity_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                WITH e
                MATCH (d:Document)-[:CONTAINS]->(e)
                RETURN d.content as content,
                       d.title as title,
                       collect(distinct e.name) as entities
                LIMIT 5
            """
            entity_results = self.graph.run(entity_query, query=query_text).data()
            self.logger.debug(f"Entity query results: {entity_results}")

            # Format response
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