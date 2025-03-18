import logging
from py2neo import Graph
from urllib.parse import urlparse
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    # Get credentials from environment
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    logger.info(f"Testing Neo4j connection with URI: {neo4j_uri}")
    logger.info(f"Username: {neo4j_user}")
    
    try:
        # Parse the URI
        uri = urlparse(neo4j_uri)
        if uri.scheme == 'neo4j+s':
            # For AuraDB, use bolt+s://
            netloc = uri.netloc
            bolt_uri = f"bolt+s://{netloc}"
            logger.info(f"Converting AuraDB URI to: {bolt_uri}")
        else:
            bolt_uri = neo4j_uri
            logger.info(f"Using original URI: {bolt_uri}")

        # Attempt connection
        logger.info("Attempting to connect to Neo4j...")
        graph = Graph(bolt_uri, auth=(neo4j_user, neo4j_password))
        
        # Test connection with a simple query
        result = graph.run("RETURN 1 as test").data()
        logger.info("Connection successful!")
        logger.info(f"Test query result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_neo4j_connection()
