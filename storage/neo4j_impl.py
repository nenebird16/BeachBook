import os
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from py2neo import Graph
from .base import GraphDatabaseInterface

logger = logging.getLogger(__name__)

class Neo4jDatabase(GraphDatabaseInterface):
    """Neo4j implementation of the graph database interface"""

    def __init__(self):
        """Initialize the database connection details"""
        self.graph = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            # Get credentials from environment
            uri = os.environ.get("NEO4J_URI")
            username = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            if not all([uri, username, password]):
                self.logger.error("Missing Neo4j credentials")
                return False

            # Parse URI to get hostname
            parsed_uri = urlparse(uri)
            host = parsed_uri.hostname
            port = 7687  # Default Neo4j bolt port

            # Create direct bolt connection
            bolt_uri = f"bolt+s://{host}:{port}"
            self.logger.info(f"Connecting to Neo4j at: {bolt_uri}")

            # Initialize database connection
            self.graph = Graph(bolt_uri, auth=(username, password))

            # Test connection
            result = self.graph.run("MATCH (n) RETURN count(n) as count LIMIT 1").data()
            node_count = result[0]['count'] if result else 0
            self.logger.info(f"Successfully connected to Neo4j database. Found {node_count} nodes.")

            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node with given label and properties"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            node = self.graph.run(
                "CREATE (n:{}) SET n = $props RETURN n".format(label),
                props=properties
            ).evaluate()
            return dict(node) if node else {}
        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            raise

    def create_relationship(self, start_node_id: int, end_node_id: int,
                          relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            result = self.graph.run("""
                MATCH (start), (end)
                WHERE ID(start) = $start_id AND ID(end) = $end_id
                CREATE (start)-[r:$rel_type $props]->(end)
                RETURN r
                """,
                start_id=start_node_id,
                end_id=end_node_id,
                rel_type=relationship_type,
                props=properties or {}
            )
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            raise

    def query(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            result = self.graph.run(query_string, **(params or {}))
            return result.data()
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def get_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            result = self.graph.run(
                "MATCH (n) WHERE ID(n) = $node_id RETURN n",
                node_id=node_id
            ).data()
            return dict(result[0]['n']) if result else None
        except Exception as e:
            self.logger.error(f"Error fetching node by ID: {str(e)}")
            raise