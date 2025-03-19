import os
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from neo4j import GraphDatabase, Driver
from .base import GraphDatabaseInterface

logger = logging.getLogger(__name__)

class Neo4jDatabase(GraphDatabaseInterface):
    """Neo4j implementation of the graph database interface"""

    def __init__(self):
        """Initialize the database connection details"""
        self.driver = None
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

            # Parse URI for connection details
            parsed_uri = urlparse(uri)
            self.logger.debug("Connecting to Neo4j database:")
            self.logger.debug(f"Original URI scheme: {parsed_uri.scheme}")
            self.logger.debug(f"Host: {parsed_uri.hostname}")
            self.logger.debug(f"Port: {parsed_uri.port or 7687}")

            # Map Neo4j protocols to bolt protocols
            protocol_mapping = {
                'neo4j': 'bolt',
                'neo4j+s': 'bolt+s',
                'neo4j+ssc': 'bolt+ssc'
            }

            # Convert protocol if needed, defaulting to original if not in mapping
            if parsed_uri.scheme in protocol_mapping:
                scheme = protocol_mapping[parsed_uri.scheme]
                connection_uri = f"{scheme}://{parsed_uri.hostname}:{parsed_uri.port or 7687}"
            else:
                connection_uri = uri

            self.logger.info(f"Connecting using URI: {connection_uri}")

            # Create the driver instance
            self.driver = GraphDatabase.driver(
                connection_uri,
                auth=(username, password)
            )

            # Verify connection with a simple query
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test").single()
                self.logger.debug(f"Connection test result: {result}")

            self.logger.info("Successfully connected to Neo4j")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Connection URI attempted: {connection_uri}")
            return False

    def query(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query"""
        if not self.driver:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Executing query: {query_string}")
            self.logger.debug(f"Query parameters: {params}")

            with self.driver.session() as session:
                result = session.run(query_string, parameters=params or {})
                data = [dict(record) for record in result]

            self.logger.debug(f"Query returned {len(data)} results")
            return data

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node with given label and properties"""
        if not self.driver:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Creating node with label: {label}")
            self.logger.debug(f"Node properties: {properties}")

            query = f"CREATE (n:{label}) SET n = $props RETURN n"
            with self.driver.session() as session:
                result = session.run(query, props=properties).single()
                return dict(result['n']) if result else {}

        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            raise

    def create_relationship(self, start_node_id: int, end_node_id: int,
                          relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes"""
        if not self.driver:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Creating relationship: ({start_node_id})-[:{relationship_type}]->({end_node_id})")
            self.logger.debug(f"Relationship properties: {properties}")

            query = """
            MATCH (start), (end)
            WHERE ID(start) = $start_id AND ID(end) = $end_id
            CREATE (start)-[r:$rel_type]->(end)
            SET r = $props
            RETURN r
            """

            with self.driver.session() as session:
                result = session.run(
                    query,
                    start_id=start_node_id,
                    end_id=end_node_id,
                    rel_type=relationship_type,
                    props=properties or {}
                ).single()
                return bool(result)

        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            raise

    def get_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID"""
        if not self.driver:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Fetching node with ID: {node_id}")

            query = "MATCH (n) WHERE ID(n) = $node_id RETURN n"
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id).single()
                return dict(result['n']) if result else None

        except Exception as e:
            self.logger.error(f"Error fetching node by ID: {str(e)}")
            raise