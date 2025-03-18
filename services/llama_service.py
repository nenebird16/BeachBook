from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage.storage_context import StorageContext
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

            # Define Cypher query templates for graph traversal
            cypher_queries = {
                "graph_rag_query": """
                    MATCH (d:Document)
                    WHERE d.content CONTAINS $query
                    WITH d, score() as score
                    MATCH (d)-[r:CONTAINS]->(e:Entity)
                    RETURN d.content as content, 
                           collect(distinct e.name) as related_entities,
                           score
                    ORDER BY score DESC
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

            # Create storage context with graph store
            self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            self.logger.info("Successfully initialized Neo4j graph store with RAG templates")

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j graph store: {str(e)}")
            raise

    def process_document(self, content):
        """Process document content and create index"""
        try:
            self.logger.info("Processing document with LlamaIndex")
            documents = [Document(text=content)]

            # Create vector index with graph store context
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context
            )
            self.logger.info("Successfully processed document and created vector index")
            return self.index
        except Exception as e:
            self.logger.error(f"Error processing document with LlamaIndex: {str(e)}")
            raise

    def process_query(self, query_text):
        """Process a query using the Graph RAG pipeline"""
        try:
            self.logger.info(f"Processing query: {query_text}")

            # Check if index exists and log its status
            has_index = hasattr(self, 'index')
            self.logger.debug(f"Vector index exists: {has_index}")

            if not has_index:
                self.logger.warning("No vector index found - document needs to be uploaded first")
                return "Please upload a document first before querying."

            # Try to verify document existence in Neo4j
            try:
                doc_check = self.graph_store.raw_query(
                    query_text="MATCH (d:Document) RETURN count(d) as count",
                    parameters={}
                )
                doc_count = doc_check[0]['count'] if doc_check else 0
                self.logger.debug(f"Number of documents in Neo4j: {doc_count}")

                if doc_count == 0:
                    self.logger.warning("No documents found in Neo4j graph")
                    return "No documents found in the knowledge graph. Please upload a document first."
            except Exception as e:
                self.logger.error(f"Error checking Neo4j documents: {str(e)}")

            # Execute graph RAG query
            result = self.graph_store.raw_query(
                query_text=query_text,
                query_type="graph_rag_query",
                parameters={"query": query_text}
            )
            self.logger.debug(f"Graph query result: {result}")

            # Create hybrid retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3
            )

            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever
            )

            # Get vector store response
            response = query_engine.query(query_text)

            # Combine responses
            combined_response = str(response)

            # Add graph context if available
            if result and len(result) > 0:
                combined_response += "\n\nGraph Context:\n"
                for idx, row in enumerate(result, 1):
                    combined_response += f"{idx}. Document content: {row['content'][:200]}...\n"
                    if row['related_entities']:
                        combined_response += f"   Related entities: {', '.join(row['related_entities'])}\n"

            return combined_response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise