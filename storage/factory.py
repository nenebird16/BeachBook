import os
import logging
from typing import Optional
from .base import GraphDatabaseInterface, ObjectStorageInterface
from .neo4j_impl import Neo4jDatabase
from .replit_storage_impl import ReplitObjectStorage

logger = logging.getLogger(__name__)

class StorageFactory:
    """Factory class for creating storage implementations"""

    @staticmethod
    def create_graph_database(db_type: str = "neo4j") -> Optional[GraphDatabaseInterface]:
        """Create and return a graph database implementation"""
        logger.info(f"Creating graph database implementation: {db_type}")

        if db_type == "neo4j":
            # Get credentials from environment
            uri = os.environ.get("NEO4J_URI")
            username = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            # Log credential presence (not values)
            logger.debug(f"NEO4J_URI present: {bool(uri)}")
            logger.debug(f"NEO4J_USER present: {bool(username)}")
            logger.debug(f"NEO4J_PASSWORD present: {bool(password)}")

            if not all([uri, username, password]):
                logger.error("Missing Neo4j credentials")
                return None

            try:
                logger.info("Attempting to create Neo4j database connection...")
                db = Neo4jDatabase(
                    uri=uri,
                    username=username,
                    password=password
                )
                logger.info("Neo4j database instance created, attempting connection...")
                db.connect()
                logger.info("Successfully connected to Neo4j database")
                return db
            except Exception as e:
                logger.error(f"Failed to create Neo4j database: {str(e)}")
                return None
        else:
            raise ValueError(f"Unsupported graph database type: {db_type}")

    @staticmethod
    def create_object_storage(storage_type: str = "replit",
                           bucket_name: Optional[str] = None) -> Optional[ObjectStorageInterface]:
        """Create and return an object storage implementation"""
        logger.info(f"Creating object storage implementation: {storage_type}")

        if storage_type == "replit":
            try:
                storage = ReplitObjectStorage(bucket_name or "default")
                storage.connect()
                return storage
            except Exception as e:
                logger.error(f"Failed to create Replit storage: {str(e)}")
                return None
        else:
            raise ValueError(f"Unsupported object storage type: {storage_type}")