import os
# Remove NEO4J_URI from environment before importing py2neo
original_uri = os.environ.pop('NEO4J_URI', None)

from py2neo import Graph, Node, Relationship, ConnectionProfile
import logging
from urllib.parse import urlparse
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class GraphService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                raise ValueError("Neo4j credentials not properly configured")

            # Parse the URI and extract hostname for AuraDB
            uri = urlparse(NEO4J_URI)
            self.logger.debug(f"Original URI scheme: {uri.scheme}")
            self.logger.debug(f"Original URI netloc: {uri.netloc}")

            try:
                # Create connection profile
                if uri.scheme == 'neo4j+s':
                    # AuraDB connection settings
                    profile = ConnectionProfile(
                        scheme="bolt+s",
                        host=uri.netloc,
                        port=7687,  # Default AuraDB port
                        secure=True,
                        user=NEO4J_USER,
                        password=NEO4J_PASSWORD
                    )
                    self.logger.info("Using AuraDB connection format")
                else:
                    profile = ConnectionProfile(
                        uri=NEO4J_URI,
                        user=NEO4J_USER,
                        password=NEO4J_PASSWORD
                    )
                    self.logger.info("Using standard Neo4j connection format")

                self.logger.debug(f"Connection profile configured: {profile.scheme}://{profile.host}")

                # Initialize Graph with the connection profile
                self.graph = Graph(profile=profile)

                # Verify connection
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

    def create_entity_node(self, entity_info, doc_node):
        """Create an entity node and link it to the document"""
        try:
            # Create entity node
            entity_node = Node("Entity",
                            name=entity_info['name'],
                            type=entity_info['type'])

            # Create relationship between document and entity
            relationship = Relationship(doc_node, "CONTAINS", entity_node)

            # Create both node and relationship in a single transaction
            self.graph.create(relationship)
            self.logger.info(f"Created entity node: {entity_info['name']} ({entity_info['type']})")

            return entity_node

        except Exception as e:
            self.logger.error(f"Error creating entity node: {str(e)}")
            raise

    def create_visual_element_node(self, element_info, doc_node):
        """Create a visual element node and link it to the document"""
        try:
            # Create visual element node
            visual_node = Node("VisualElement",
                             name=element_info['name'],
                             type=element_info['type'])

            # Create relationship to document
            relationship = Relationship(doc_node, "CONTAINS", visual_node)

            # Create both in a single transaction
            self.graph.create(relationship)
            self.logger.info(f"Created visual element node: {element_info['name']}")

            return visual_node

        except Exception as e:
            self.logger.error(f"Error creating visual element node: {str(e)}")
            raise

    def create_relationship(self, source_name, source_type, target_name, target_type, 
                          rel_type, properties=None):
        """Create a relationship between two existing nodes"""
        try:
            # Find source and target nodes
            source_query = """
            MATCH (n:Entity {name: $name, type: $type})
            RETURN n LIMIT 1
            """
            source_result = self.graph.run(source_query, 
                                         name=source_name, 
                                         type=source_type).data()

            target_query = """
            MATCH (n:Entity {name: $name, type: $type})
            RETURN n LIMIT 1
            """
            target_result = self.graph.run(target_query, 
                                         name=target_name, 
                                         type=target_type).data()

            if not source_result or not target_result:
                raise ValueError(f"Could not find nodes for relationship: {source_name} -> {target_name}")

            # Create relationship with properties
            create_rel_query = f"""
            MATCH (s:Entity {{name: $source_name, type: $source_type}})
            MATCH (t:Entity {{name: $target_name, type: $target_type}})
            CREATE (s)-[r:{rel_type}]->(t)
            SET r += $props
            RETURN r
            """

            self.graph.run(create_rel_query,
                          source_name=source_name,
                          source_type=source_type,
                          target_name=target_name,
                          target_type=target_type,
                          props=properties or {})

            self.logger.info(f"Created relationship: {source_name} -[{rel_type}]-> {target_name}")

        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            raise