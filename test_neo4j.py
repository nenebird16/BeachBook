import os
# Remove NEO4J_URI from environment before importing py2neo
original_uri = os.environ.pop('NEO4J_URI', None)

import logging
from py2neo import Graph, ConnectionProfile
from urllib.parse import urlparse
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_connections():
    try:
        # Get credentials from environment (NEO4J_URI was already removed)
        neo4j_user = os.environ.get("NEO4J_USER")
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        openai_key = os.environ.get("OPENAI_API_KEY")

        # Verify environment variables
        logger.debug(f"NEO4J_USER present: {'Yes' if neo4j_user else 'No'}")
        logger.debug(f"NEO4J_PASSWORD present: {'Yes' if neo4j_password else 'No'}")
        logger.debug(f"OPENAI_API_KEY present: {'Yes' if openai_key else 'No'}")

        if not all([original_uri, neo4j_user, neo4j_password]):
            raise ValueError("Neo4j credentials not properly configured")

        # Parse URI for AuraDB connection
        uri = urlparse(original_uri)
        logger.debug(f"Original URI scheme: {uri.scheme}")
        logger.debug(f"Original URI netloc: {uri.netloc}")

        try:
            # Create connection profile for AuraDB
            if uri.scheme == 'neo4j+s':
                profile = ConnectionProfile(
                    scheme="bolt+s",
                    host=uri.netloc,
                    port=7687,  # Default AuraDB port
                    secure=True,
                    user=neo4j_user,
                    password=neo4j_password
                )
                logger.info("Using AuraDB connection format")
            else:
                profile = ConnectionProfile(
                    uri=original_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                logger.info("Using standard Neo4j connection format")

            logger.debug(f"Connection profile configured: {profile.scheme}://{profile.host}")

            # Test direct Neo4j connection
            logger.info("Testing direct Neo4j connection...")
            try:
                graph = Graph(profile=profile)
                result = graph.run("RETURN 1 as test").data()
                logger.info("✓ Direct Neo4j connection successful")
                logger.debug(f"Test query result: {result}")

            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {str(e)}")
                raise

            # Test LlamaIndex connection
            logger.info("Testing LlamaIndex connection...")
            try:
                Settings.llm_api_key = openai_key

                # Configure LlamaIndex connection
                if uri.scheme == 'neo4j+s':
                    url = f"bolt+s://{uri.netloc}"
                else:
                    url = original_uri

                logger.debug(f"LlamaIndex connection URI: {url}")
                graph_store = Neo4jGraphStore(
                    username=neo4j_user,
                    password=neo4j_password,
                    url=url,
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
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_connections()