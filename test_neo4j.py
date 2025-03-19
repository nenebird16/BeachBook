import os
import logging
from storage.neo4j_impl import Neo4jDatabase
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """Test Neo4j database connection"""
    try:
        # Initialize Neo4j database
        db = Neo4jDatabase()
        logger.info("Testing Neo4j connection...")

        # Try to connect
        if db.connect():
            logger.info("✓ Successfully connected to Neo4j")

            # Test query execution
            try:
                # Run a simple test query
                result = db.query("MATCH (n) RETURN count(n) as count")
                logger.info(f"Total nodes in database: {result[0]['count'] if result else 0}")

                # Check for Player nodes specifically
                player_result = db.query("""
                    MATCH (p:Player) 
                    RETURN count(p) as count
                """)
                logger.info(f"Total Player nodes: {player_result[0]['count'] if player_result else 0}")

                return True

            except Exception as e:
                logger.error(f"Query execution failed: {str(e)}")
                return False
        else:
            logger.error("Failed to connect to Neo4j")
            return False

    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_neo4j_connection()