import os
import logging
from typing import Optional
from py2neo import Graph, ConnectionProfile
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class StorageFactory:
    """Factory class for creating storage implementations"""

    @staticmethod
    def create_graph_database(db_type: str = "neo4j") -> Optional['GraphDatabaseInterface']:
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
                # Parse URI and map to py2neo compatible scheme
                parsed_uri = urlparse(uri)
                scheme_mapping = {
                    'neo4j': 'bolt',
                    'neo4j+s': 'bolt+s',
                    'bolt': 'bolt',
                    'bolt+s': 'bolt+s'
                }

                original_scheme = parsed_uri.scheme
                scheme = scheme_mapping.get(original_scheme, 'bolt')

                logger.info(f"Creating Neo4j connection with scheme: {scheme} (mapped from {original_scheme})")
                logger.debug(f"Host: {parsed_uri.hostname}, Port: {parsed_uri.port or 7687}")

                # Create connection profile
                profile = ConnectionProfile(
                    scheme=scheme,
                    host=parsed_uri.hostname,
                    port=parsed_uri.port or 7687,
                    secure=scheme.endswith('+s'),
                    user=username,
                    password=password
                )

                # Initialize database connection
                graph = Graph(profile=profile)

                # Test connection
                result = graph.run("MATCH (n) RETURN count(n) as count LIMIT 1").data()
                node_count = result[0]['count'] if result else 0
                logger.info(f"Successfully connected to Neo4j database. Found {node_count} nodes.")

                return graph

            except Exception as e:
                logger.error(f"Failed to create Neo4j database: {str(e)}")
                return None
        else:
            raise ValueError(f"Unsupported graph database type: {db_type}")

    @staticmethod
    def create_object_storage(storage_type: str = "replit",
                           bucket_name: Optional[str] = None) -> Optional['ObjectStorageInterface']:
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