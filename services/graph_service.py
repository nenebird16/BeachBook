from py2neo import Graph, Node, Relationship
import logging
from urllib.parse import urlparse, urlunparse
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class GraphService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Parse the URI and convert neo4j protocol to bolt
            uri = urlparse(NEO4J_URI)
            if uri.scheme == 'neo4j+s':
                # For neo4j+s://, use bolt+s://
                scheme = 'bolt+s'
                bolt_uri = urlunparse((scheme, uri.netloc, uri.path, uri.params, uri.query, uri.fragment))
                self.logger.info(f"Converting Neo4j URI from {NEO4J_URI} to {bolt_uri}")
            elif uri.scheme == 'neo4j':
                # For neo4j://, use bolt://
                scheme = 'bolt'
                bolt_uri = urlunparse((scheme, uri.netloc, uri.path, uri.params, uri.query, uri.fragment))
                self.logger.info(f"Converting Neo4j URI from {NEO4J_URI} to {bolt_uri}")
            else:
                bolt_uri = NEO4J_URI
                self.logger.info(f"Using original URI: {bolt_uri}")

            self.logger.debug(f"Attempting to connect to Neo4j with user: {NEO4J_USER}")
            self.graph = Graph(bolt_uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
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