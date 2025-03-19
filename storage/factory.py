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
    def create_graph_database(db_type: str = "neo4j") -> GraphDatabaseInterface:
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

            try:
                return Neo4jDatabase(
                    uri=uri or "",
                    username=username or "",
                    password=password or ""
                )
            except ValueError as e:
                logger.error(f"Failed to create Neo4j database: {str(e)}")
                raise ValueError(
                    "Neo4j credentials not properly configured. Please ensure NEO4J_URI, "
                    "NEO4J_USER, and NEO4J_PASSWORD environment variables are set."
                ) from e
        else:
            raise ValueError(f"Unsupported graph database type: {db_type}")

    @staticmethod
    def create_object_storage(storage_type: str = "replit",
                           bucket_name: Optional[str] = None) -> ObjectStorageInterface:
        """Create and return an object storage implementation"""
        logger.info(f"Creating object storage implementation: {storage_type}")

        if storage_type == "replit":
            try:
                return ReplitObjectStorage(bucket_name or "default")
            except Exception as e:
                logger.error(f"Failed to create Replit storage: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported object storage type: {storage_type}")