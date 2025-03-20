import os
import logging
from storage.neo4j_impl import Neo4jDatabase
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_neo4j_data():
    """Test Neo4j database connection and inspect data"""
    try:
        # Initialize Neo4j database
        db = Neo4jDatabase()
        logger.info("Testing Neo4j connection and data...")

        # Try to connect
        if db.connect():
            logger.info("✓ Successfully connected to Neo4j")

            # Check all node labels
            labels_query = """
            CALL db.labels() YIELD label
            RETURN collect(label) as labels
            """
            labels_result = db.query(labels_query)
            logger.info(f"Available node labels: {labels_result[0]['labels'] if labels_result else []}")

            # Sample nodes for each label
            node_samples_query = """
            MATCH (n)
            WITH labels(n) as labels, n
            RETURN DISTINCT 
                labels[0] as label,
                n.name as name,
                keys(n) as properties
            LIMIT 5
            """
            samples = db.query(node_samples_query)
            logger.info("Sample nodes:")
            for sample in samples:
                logger.info(f"Label: {sample['label']}")
                logger.info(f"Name: {sample['name']}")
                logger.info(f"Properties: {sample['properties']}")
                logger.info("---")

            return True

        else:
            logger.error("Failed to connect to Neo4j")
            return False

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_neo4j_data()