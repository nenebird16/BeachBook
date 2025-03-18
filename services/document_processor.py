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
            # Create document info
            doc_info = {
                'title': file.filename,
                'timestamp': datetime.now().isoformat()
            }

            # Handle different file types
            file_content = None
            if file.filename.endswith('.txt'):
                # For text files, decode as UTF-8
                file_content = file.read().decode('utf-8')
            elif file.filename.endswith(('.pdf', '.doc', '.docx')):
                # For binary files, read as bytes
                file_content = file.read()
                # TODO: Add proper binary file processing here
                # For now, return an informative error
                raise NotImplementedError(
                    "Binary file processing is not yet implemented. "
                    "Currently supported formats: .txt"
                )
            else:
                raise ValueError(
                    "Unsupported file type. "
                    "Currently supported formats: .txt"
                )

            doc_info['content'] = file_content

            # Process with LlamaIndex
            self.llama_service.process_document(file_content)

            # Create document node in Neo4j
            doc_node = self.graph_service.create_document_node(doc_info)

            # Extract and create entity relationships
            self.create_basic_entities(doc_node, file_content)

            return doc_info

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def create_basic_entities(self, doc_node, content):
        """Create basic entities from document content"""
        # This is a simplified implementation
        words = content.split()
        for word in set(words):
            if len(word) > 5:  # Simple filter for demonstration
                entity_info = {
                    'name': word,
                    'type': 'keyword'
                }
                self.graph_service.create_entity_relationship(doc_node, entity_info)