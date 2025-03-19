import os
import logging
from py2neo import Graph
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_connections():
    try:
        # Get credentials from environment
        neo4j_user = os.environ.get("NEO4J_USER")
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        uri = os.environ.get("NEO4J_URI")

        # Verify environment variables
        logger.debug(f"NEO4J_USER present: {'Yes' if neo4j_user else 'No'}")
        logger.debug(f"NEO4J_PASSWORD present: {'Yes' if neo4j_password else 'No'}")

        if not all([uri, neo4j_user, neo4j_password]):
            raise ValueError("Neo4j credentials not properly configured")

        # Parse URI to get hostname
        parsed_uri = urlparse(uri)
        host = parsed_uri.hostname
        port = 7687  # Default Neo4j bolt port

        # Create direct bolt connection string
        bolt_uri = f"bolt+s://{host}:{port}"
        logger.info(f"Connecting to Neo4j at: {bolt_uri}")

        # Test direct Neo4j connection
        logger.info("Testing direct Neo4j connection...")
        try:
            graph = Graph(bolt_uri, auth=(neo4j_user, neo4j_password))

            # Test basic connectivity
            result = graph.run("RETURN 1 as test").data()
            logger.info("✓ Direct Neo4j connection successful")
            logger.debug(f"Test query result: {result}")

            # Check for nodes and their properties
            node_count = graph.run("MATCH (n) RETURN count(n) as count").data()
            logger.info(f"Total nodes in database: {node_count[0]['count'] if node_count else 0}")

            # Check for nodes with content property
            content_nodes = graph.run("""
                MATCH (n) 
                WHERE exists(n.content)
                RETURN count(n) as count, 
                       labels(n)[0] as label
                """).data()

            if content_nodes:
                logger.info("Nodes with content property:")
                for result in content_nodes:
                    logger.info(f"- {result['label']}: {result['count']} nodes")
            else:
                logger.warning("No nodes found with 'content' property")

            # Sample node properties
            sample_nodes = graph.run("""
                MATCH (n) 
                RETURN labels(n) as labels, 
                       properties(n) as props 
                LIMIT 1
                """).data()

            if sample_nodes:
                logger.info("Sample node structure:")
                for node in sample_nodes:
                    logger.info(f"Labels: {node['labels']}")
                    logger.info(f"Properties: {node['props']}")
            else:
                logger.warning("No nodes found in database")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return False

    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_connections()