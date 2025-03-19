import os
from typing import Optional
from .base import GraphDatabaseInterface, ObjectStorageInterface
from .neo4j_impl import Neo4jDatabase
from .replit_storage_impl import ReplitObjectStorage

class StorageFactory:
    """Factory class for creating storage implementations"""
    
    @staticmethod
    def create_graph_database(db_type: str = "neo4j") -> GraphDatabaseInterface:
        """Create and return a graph database implementation"""
        if db_type == "neo4j":
            return Neo4jDatabase(
                uri=os.environ.get("NEO4J_URI"),
                username=os.environ.get("NEO4J_USER"),
                password=os.environ.get("NEO4J_PASSWORD")
            )
        else:
            raise ValueError(f"Unsupported graph database type: {db_type}")
    
    @staticmethod
    def create_object_storage(storage_type: str = "replit",
                            bucket_name: Optional[str] = None) -> ObjectStorageInterface:
        """Create and return an object storage implementation"""
        if storage_type == "replit":
            return ReplitObjectStorage(bucket_name or "default")
        else:
            raise ValueError(f"Unsupported object storage type: {storage_type}")
