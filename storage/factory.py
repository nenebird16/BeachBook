import os
import logging
from typing import Optional
from .base import GraphDatabaseInterface, ObjectStorageInterface
from .neo4j_impl import Neo4jDatabase

logger = logging.getLogger(__name__)

class StorageFactory:
    """Factory class for creating storage implementations"""

    @staticmethod
    def create_graph_database(db_type: str = "neo4j") -> Optional[GraphDatabaseInterface]:
        """Create and return a graph database implementation"""
        logger.info(f"Creating graph database implementation: {db_type}")

        if db_type == "neo4j":
            try:
                # Create Neo4j database instance
                db = Neo4jDatabase()

                # Attempt connection
                if db.connect():
                    logger.info("Successfully created and connected Neo4j database")
                    return db
                else:
                    logger.error("Failed to connect to Neo4j database")
                    return None

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
                from replit import db
                return db
            except Exception as e:
                logger.error(f"Failed to create Replit storage: {str(e)}")
                return None
        else:
            raise ValueError(f"Unsupported object storage type: {storage_type}")