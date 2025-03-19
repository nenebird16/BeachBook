import os
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from py2neo import Graph, Node, Relationship, ConnectionProfile
from .base import GraphDatabaseInterface

logger = logging.getLogger(__name__)

class Neo4jDatabase(GraphDatabaseInterface):
    """Neo4j implementation of the graph database interface"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.graph = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        try:
            if not all([self.uri, self.username, self.password]):
                raise ValueError("Neo4j credentials not properly configured")

            # Parse URI for AuraDB
            uri = urlparse(self.uri)
            self.logger.debug(f"Original URI scheme: {uri.scheme}")
            self.logger.debug(f"Original URI netloc: {uri.netloc}")

            # Initialize direct Neo4j connection
            profile = ConnectionProfile(
                scheme="bolt+s" if uri.scheme == 'neo4j+s' else uri.scheme,
                host=uri.netloc,
                port=7687,
                secure=True if uri.scheme == 'neo4j+s' else False,
                user=self.username,
                password=self.password
            )
            self.graph = Graph(profile=profile)
            
            # Test connection
            result = self.graph.run("RETURN 1 as test").data()
            self.logger.info("Successfully connected to Neo4j database")
            self.logger.debug(f"Test query result: {result}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        try:
            node = Node(label, **properties)
            self.graph.create(node)
            return dict(node)
        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            raise
    
    def create_relationship(self, start_node_id: int, end_node_id: int,
                          relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        try:
            query = """
            MATCH (start), (end)
            WHERE ID(start) = $start_id AND ID(end) = $end_id
            CREATE (start)-[r:$rel_type $props]->(end)
            RETURN r
            """
            result = self.graph.run(
                query,
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
        try:
            result = self.graph.run(query_string, **(params or {}))
            return result.data()
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        try:
            query = "MATCH (n) WHERE ID(n) = $node_id RETURN n"
            result = self.graph.run(query, node_id=node_id).data()
            return dict(result[0]['n']) if result else None
        except Exception as e:
            self.logger.error(f"Error fetching node by ID: {str(e)}")
            raise
