import os
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Store original URI
original_uri = os.environ.get('NEO4J_URI')
# Remove NEO4J_URI from environment before importing py2neo
os.environ.pop('NEO4J_URI', None)

from neo4j import GraphDatabase, Driver
from py2neo import Graph, ConnectionProfile
from .base import GraphDatabaseInterface

# Restore original URI after py2neo import
if original_uri:
    os.environ['NEO4J_URI'] = original_uri

logger = logging.getLogger(__name__)

class Neo4jDatabase(GraphDatabaseInterface):
    """Neo4j implementation of the graph database interface"""

    def __init__(self):
        """Initialize the database connection details"""
        self.driver = None
        self.graph = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            # Get credentials but use stored original URI
            username = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")
            uri = original_uri  # Use the stored original URI

            if not all([uri, username, password]):
                self.logger.error("Missing Neo4j credentials:")
                self.logger.error(f"URI present: {'Yes' if uri else 'No'}")
                self.logger.error(f"Username present: {'Yes' if username else 'No'}")
                self.logger.error(f"Password present: {'Yes' if password else 'No'}")
                return False

            # Parse URI for connection details
            parsed_uri = urlparse(uri)
            self.logger.debug("Connecting to Neo4j database:")
            self.logger.debug(f"Original URI scheme: {parsed_uri.scheme}")
            self.logger.debug(f"Original URI netloc: {parsed_uri.netloc}")

            try:
                # Handle AuraDB connections (neo4j+s scheme)
                if parsed_uri.scheme == 'neo4j+s':
                    bolt_uri = f"bolt+s://{parsed_uri.netloc}"
                    self.logger.info(f"Using AuraDB connection format: {bolt_uri}")
                else:
                    bolt_uri = f"bolt://{parsed_uri.netloc}"
                    self.logger.info(f"Using standard connection format: {bolt_uri}")

                # Create connection profile with bolt URI
                profile = ConnectionProfile(
                    uri=bolt_uri,
                    user=username,
                    password=password
                )
                self.logger.debug(f"Connection profile configured: {bolt_uri}")

                # Initialize Graph with the connection profile
                self.graph = Graph(profile=profile)

                # Verify connection with a test query
                result = self.graph.run("RETURN 1 as test").data()
                self.logger.info("Successfully connected to Neo4j database")
                self.logger.debug(f"Test query result: {result}")

                return True

            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Error establishing database connection: {str(e)}")
            return False

    def query(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Executing query: {query_string}")
            self.logger.debug(f"Query parameters: {params}")

            result = self.graph.run(query_string, parameters=params or {})
            data = [dict(record) for record in result]

            self.logger.debug(f"Query returned {len(data)} results")
            return data

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def create_document_node(self, doc_info: Dict[str, Any]) -> Any:
        """Create a document node with its metadata"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Creating document node with info: {doc_info}")

            # Create document node
            node = self.graph.run("""
                CREATE (d:Document {
                    title: $title,
                    content: $content,
                    timestamp: $timestamp
                })
                RETURN d
                """,
                title=doc_info['title'],
                content=doc_info['content'],
                timestamp=doc_info['timestamp']
            ).evaluate()

            if node:
                self.logger.info(f"Created document node: {doc_info['title']}")
                return node
            else:
                raise Exception("Failed to create document node")

        except Exception as e:
            self.logger.error(f"Error creating document node: {str(e)}")
            raise

    def create_entity_node(self, entity_info: Dict[str, Any], doc_node: Any) -> Any:
        """Create an entity node and link it to the document"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Creating entity node: {entity_info}")

            # Create entity node and relationship in one transaction
            result = self.graph.run("""
                MATCH (d:Document)
                WHERE id(d) = $doc_id
                CREATE (e:Entity {name: $name, type: $type})
                CREATE (d)-[r:CONTAINS]->(e)
                RETURN e
                """,
                doc_id=doc_node.identity,
                name=entity_info['name'],
                type=entity_info['type']
            ).evaluate()

            if result:
                self.logger.info(f"Created entity node: {entity_info['name']} ({entity_info['type']})")
                return result
            else:
                raise Exception("Failed to create entity node")

        except Exception as e:
            self.logger.error(f"Error creating entity node: {str(e)}")
            raise

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node with given label and properties"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Creating node with label: {label}")
            self.logger.debug(f"Node properties: {properties}")

            query = """
            CREATE (n:$label)
            SET n = $props
            RETURN n
            """
            result = self.graph.run(
                query,
                label=label,
                props=properties
            ).evaluate()

            if result:
                self.logger.info(f"Created node with label {label}")
                return dict(result)
            else:
                raise Exception(f"Failed to create node with label {label}")

        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            raise

    def create_relationship(self, start_node_id: int, end_node_id: int,
                          relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes"""
        if not self.graph:
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

            result = self.graph.run(
                query,
                start_id=start_node_id,
                end_id=end_node_id,
                rel_type=relationship_type,
                props=properties or {}
            ).evaluate()

            success = result is not None
            if success:
                self.logger.info(f"Created relationship of type {relationship_type}")
            else:
                self.logger.warning("Failed to create relationship")
            return success

        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            raise

    def get_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID"""
        if not self.graph:
            raise RuntimeError("Database connection not established. Call connect() first.")

        try:
            self.logger.debug(f"Fetching node with ID: {node_id}")

            query = """
            MATCH (n)
            WHERE ID(n) = $node_id
            RETURN n
            """
            result = self.graph.run(query, node_id=node_id).evaluate()

            if result:
                return dict(result)
            return None

        except Exception as e:
            self.logger.error(f"Error fetching node by ID: {str(e)}")
            raise