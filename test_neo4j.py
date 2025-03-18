
import logging
from py2neo import Graph
from urllib.parse import urlparse
import os
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_connections():
    # Get credentials from environment
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    openai_key = os.environ.get("OPENAI_API_KEY")

    logger.info("Testing Neo4j and LlamaIndex connections...")
    
    try:
        # Test direct Neo4j connection
        uri = urlparse(neo4j_uri)
        bolt_uri = f"bolt+s://{uri.netloc}" if uri.scheme == 'neo4j+s' else neo4j_uri
        
        graph = Graph(bolt_uri, auth=(neo4j_user, neo4j_password))
        result = graph.run("RETURN 1 as test").data()
        logger.info("✓ Direct Neo4j connection successful")

        # Test LlamaIndex connection
        Settings.llm_api_key = openai_key
        graph_store = Neo4jGraphStore(
            username=neo4j_user,
            password=neo4j_password,
            url=bolt_uri,
            database="neo4j"
        )
        
        # Try to create a test document and index
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        test_doc = Document(text="Test document for connection verification")
        index = VectorStoreIndex.from_documents(
            [test_doc],
            storage_context=storage_context
        )
        logger.info("✓ LlamaIndex connection and indexing successful")
        
        return True

    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_connections()
