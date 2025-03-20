import logging
from datetime import datetime
import spacy
from typing import Dict, List
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class DocumentProcessor:
    def __init__(self, graph_service, semantic_processor=None):
        self.graph_service = graph_service
        self.semantic_processor = semantic_processor
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.info("Downloading spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Successfully loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {str(e)}")
            raise

    def process_document(self, file) -> Dict:
        """Process uploaded document and store in knowledge graph with semantic analysis"""
        try:
            # Create document info
            doc_info = {
                'title': file.filename,
                'timestamp': datetime.now().isoformat(),
                'stage': 'extracting',
                'progress': 20
            }

            # Extract file content
            self.logger.info(f"Extracting content from file: {file.filename}")
            file_content = self._extract_file_content(file)
            doc_info['content'] = file_content
            self.logger.debug(f"Successfully extracted content, length: {len(file_content)}")

            # Update progress
            doc_info['stage'] = 'processing'
            doc_info['progress'] = 40

            # Create document node in Neo4j
            self.logger.info("Creating document node in Neo4j...")
            if not self.graph_service:
                raise ValueError("Graph service not initialized")

            doc_node = self.graph_service.create_document_node(doc_info)
            self.logger.info("Document node created successfully in Neo4j")

            # Update progress
            doc_info['stage'] = 'analyzing'
            doc_info['progress'] = 60

            # Extract and create entity relationships
            self.logger.info("Creating entity relationships...")
            entities = self._extract_entities(file_content)
            self._create_entity_nodes(doc_node, entities)
            self.logger.info(f"Created {len(entities)} entity relationships")

            # Final progress update
            doc_info['stage'] = 'complete'
            doc_info['progress'] = 100

            return doc_info

        except ValueError as e:
            self.logger.error(f"Initialization error: {str(e)}")
            doc_info['stage'] = 'error'
            doc_info['error'] = f"Service initialization error: {str(e)}"
            return doc_info
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}", exc_info=True)
            doc_info['stage'] = 'error'
            doc_info['error'] = f"Processing error: {str(e)}"
            return doc_info

    def _extract_file_content(self, file) -> str:
        """Extract content from file based on file type"""
        try:
            # Handle both file-like objects and our custom FileWrapper
            if hasattr(file, 'read'):
                content = file.read()
                content_str = content.decode('utf-8') if isinstance(content, bytes) else str(content)
                if not content_str.strip():
                    raise ValueError("File is empty")
                return content_str
            elif hasattr(file, 'content'):
                if not file.content.strip():
                    raise ValueError("File is empty")
                return str(file.content)
            else:
                raise ValueError("Invalid file object provided")

        except UnicodeDecodeError as e:
            self.logger.error(f"File encoding error: {str(e)}")
            raise ValueError("File encoding not supported. Please upload a valid text file.")
        except Exception as e:
            self.logger.error(f"Error extracting file content: {str(e)}")
            raise ValueError(f"Could not read file content: {str(e)}")

    def _extract_entities(self, content: str) -> List[Dict]:
        """Extract domain-specific entities from content"""
        try:
            doc = self.nlp(content)
            entities = []

            # Extract named entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities.append({
                        'name': ent.text,
                        'type': 'Player',
                        'source': 'spacy_ner'
                    })
                else:
                    entities.append({
                        'name': ent.text,
                        'type': ent.label_,
                        'source': 'spacy_ner'
                    })

            # Extract domain-specific terms
            domain_terms = {
                'Skill': [
                    'setting', 'passing', 'blocking', 'serving', 'attacking', 'digging',
                    'defense', 'reception', 'court coverage', 'jump serve'
                ],
                'Drill': [
                    'pepper', 'queen of the court', 'mini-game', 'scrimmage', 
                    'target practice', 'blocking drill', 'defensive drill'
                ],
                'VisualElement': [
                    'ball tracking', 'peripheral vision', 'trajectory prediction',
                    'depth perception', 'pattern recognition', 'visual focus'
                ]
            }

            # Extract domain-specific terms
            for entity_type, terms in domain_terms.items():
                for term in terms:
                    if term.lower() in content.lower():
                        entities.append({
                            'name': term,
                            'type': entity_type,
                            'source': 'domain_terminology'
                        })

            self.logger.info(f"Extracted {len(entities)} entities from content")
            return entities

        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _create_entity_nodes(self, doc_node, entities: List[Dict]) -> None:
        """Create entity nodes and link them to the document"""
        try:
            if not entities:
                self.logger.warning("No entities found to create nodes")
                return

            for entity in entities:
                try:
                    # Create the entity and relationship to document
                    self.graph_service.create_entity_node(entity, doc_node)
                except Exception as e:
                    self.logger.error(f"Error creating entity node: {str(e)}")
                    continue

            self.logger.info(f"Successfully created {len(entities)} entity nodes")

        except Exception as e:
            self.logger.error(f"Error in entity node creation: {str(e)}")
            raise