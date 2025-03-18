import logging
from datetime import datetime

class DocumentProcessor:
    def __init__(self, graph_service, llama_service):
        self.graph_service = graph_service
        self.llama_service = llama_service
        self.logger = logging.getLogger(__name__)

    def process_document(self, file):
        """Process uploaded document and store in knowledge graph"""
        try:
            content = file.read().decode('utf-8')
            
            # Create document info
            doc_info = {
                'title': file.filename,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }

            # Process with LlamaIndex
            self.llama_service.process_document(content)

            # Create document node in Neo4j
            doc_node = self.graph_service.create_document_node(doc_info)

            # Extract and create entity relationships
            # This is a simplified version - in practice, you'd use NLP for entity extraction
            self.create_basic_entities(doc_node, content)

            return doc_info

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def create_basic_entities(self, doc_node, content):
        """Create basic entities from document content"""
        # This is a simplified implementation
        # In practice, you'd use NLP tools for proper entity extraction
        words = content.split()
        for word in set(words):
            if len(word) > 5:  # Simple filter for demonstration
                entity_info = {
                    'name': word,
                    'type': 'keyword'
                }
                self.graph_service.create_entity_relationship(doc_node, entity_info)
