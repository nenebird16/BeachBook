from py2neo import Graph, Node, Relationship
import logging
from urllib.parse import urlparse, urlunparse
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class GraphService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                raise ValueError("Neo4j credentials not properly configured")

            # Parse and convert the URI for AuraDB connection
            uri = urlparse(NEO4J_URI)
            self.logger.debug(f"Original URI scheme: {uri.scheme}")
            self.logger.debug(f"Original URI netloc: {uri.netloc}")

            # Convert neo4j+s to bolt+s for AuraDB
            if uri.scheme == 'neo4j+s':
                # Reconstruct the URI with bolt+s scheme
                bolt_uri = urlunparse(('bolt+s', uri.netloc, '', '', '', ''))
            elif uri.scheme == 'neo4j':
                # Standard Neo4j
                bolt_uri = urlunparse(('bolt', uri.netloc, '', '', '', ''))
            else:
                bolt_uri = NEO4J_URI

            self.logger.info(f"Converted URI scheme: {urlparse(bolt_uri).scheme}")
            self.logger.debug(f"Attempting to connect with user: {NEO4J_USER}")

            try:
                self.graph = Graph(
                    bolt_uri,
                    auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                # Test the connection
                result = self.graph.run("RETURN 1 as test").data()
                self.logger.info("Successfully connected to Neo4j database")
                self.logger.debug(f"Test query result: {result}")
            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
                raise
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphService: {str(e)}")
            raise

    def create_document_node(self, doc_info):
        """Create a node for the document with its metadata"""
        try:
            node = Node("Document",
                       title=doc_info['title'],
                       content=doc_info['content'],
                       timestamp=doc_info['timestamp'])
            self.graph.create(node)
            return node
        except Exception as e:
            self.logger.error(f"Error creating document node: {str(e)}")
            raise

    def create_entity_relationship(self, doc_node, entity_info):
        """Create entity nodes and relationships to the document"""
        try:
            entity_node = Node("Entity",
                             name=entity_info['name'],
                             type=entity_info['type'])
            relationship = Relationship(doc_node, "CONTAINS", entity_node)
            self.graph.create(relationship)
        except Exception as e:
            self.logger.error(f"Error creating entity relationship: {str(e)}")
            raise

    def get_visualization_data(self):
        """Get graph data in a format suitable for visualization"""
        try:
            query = """
            MATCH (n)-[r]->(m)
            RETURN collect(distinct {id: id(n), label: labels(n)[0], properties: properties(n)}) as nodes,
                   collect(distinct {source: id(n), target: id(m), type: type(r)}) as relationships
            """
            result = self.graph.run(query).data()[0]
            return {
                'nodes': result['nodes'],
                'links': result['relationships']
            }
        except Exception as e:
            self.logger.error(f"Error fetching graph data: {str(e)}")
            raise