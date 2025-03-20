
from services.graph_service import GraphService
import logging

logger = logging.getLogger(__name__)

def setup_test_data():
    """Create test nodes and relationships for testing GraphRAG improvements"""
    try:
        from services.graph_service import GraphService
        from services.document_processor import DocumentProcessor
        from services.semantic_processor import SemanticProcessor

        graph = GraphService()
        semantic_processor = SemanticProcessor()
        doc_processor = DocumentProcessor(graph, semantic_processor)
        
        # Sample documents
        docs = [
            {
                'title': 'Test Movement Guide',
                'content': 'Movement patterns require proper muscle length and tension. Short muscles limit range of motion.',
                'timestamp': '2025-03-20T00:00:00'
            },
            {
                'title': 'Training Concepts',
                'content': 'Muscle length-tension relationships affect performance. Optimal muscle length is key for power.',
                'timestamp': '2025-03-20T00:00:00'
            }
        ]
        
        # Process each document
        for doc in docs:
            doc_node = graph.create_document_node(doc)
            entities = doc_processor._extract_entities(doc['content'])
            for entity in entities:
                graph.create_entity_node(entity, doc_node)
            
            # Add embeddings
            doc['embedding'] = semantic_processor.get_text_embedding(doc['content'])

        logger.info("Test data setup complete")
        return True
        
        # Create document nodes
        doc_nodes = []
        for doc in docs:
            node = graph.create_document_node(doc)
            doc_nodes.append(node)
            
        # Create entity nodes with relationships
        entities = [
            {'name': 'muscle length', 'type': 'Concept'},
            {'name': 'tension', 'type': 'Concept'},
            {'name': 'power output', 'type': 'Metric'},
            {'name': 'range of motion', 'type': 'Metric'}
        ]
        
        for doc_node in doc_nodes:
            for entity in entities:
                graph.create_entity_node(entity, doc_node)
                
        logger.info("Test data setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up test data: {str(e)}")
        return False
