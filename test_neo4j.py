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
    neo4j_user = os.environ.get("NEO4J_USER")  # Using NEO4J_USER consistently
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    openai_key = os.environ.get("OPENAI_API_KEY")

    logger.info("Testing Neo4j and LlamaIndex connections...")

    try:
        # Verify environment variables
        logger.debug(f"NEO4J_URI present: {'Yes' if neo4j_uri else 'No'}")
        logger.debug(f"NEO4J_USER present: {'Yes' if neo4j_user else 'No'}")
        logger.debug(f"NEO4J_PASSWORD present: {'Yes' if neo4j_password else 'No'}")
        logger.debug(f"OPENAI_API_KEY present: {'Yes' if openai_key else 'No'}")

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError("Neo4j credentials not properly configured")

        # Parse URI for AuraDB connection
        uri = urlparse(neo4j_uri)
        logger.debug(f"Original URI scheme: {uri.scheme}")
        logger.debug(f"Original URI netloc: {uri.netloc}")

        # For AuraDB, construct the bolt URL directly
        if uri.scheme == 'neo4j+s':
            bolt_uri = f"bolt+s://{uri.netloc}"
            logger.info("Using AuraDB connection format")
        else:
            bolt_uri = neo4j_uri
            logger.info("Using standard Neo4j connection format")

        logger.debug(f"Final connection URI (without credentials): {bolt_uri}")

        # Test direct Neo4j connection
        logger.info("Testing direct Neo4j connection...")
        try:
            # Temporarily remove NEO4J_URI from environment
            original_uri = os.environ.pop('NEO4J_URI', None)

            graph = Graph(
                bolt_uri,
                auth=(neo4j_user, neo4j_password)
            )

            # Restore original URI if it existed
            if original_uri:
                os.environ['NEO4J_URI'] = original_uri

            result = graph.run("RETURN 1 as test").data()
            logger.info("✓ Direct Neo4j connection successful")
            logger.debug(f"Test query result: {result}")
        except Exception as e:
            # Restore original URI in case of error
            if original_uri:
                os.environ['NEO4J_URI'] = original_uri
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

        # Test LlamaIndex connection
        logger.info("Testing LlamaIndex connection...")
        try:
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
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex: {str(e)}")
            raise

        return True

    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_connections()