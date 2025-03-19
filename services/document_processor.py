import logging
from datetime import datetime
import spacy
from typing import Dict, List
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

class DocumentProcessor:
    def __init__(self, graph_service, llama_service, semantic_processor=None):
        self.graph_service = graph_service
        self.llama_service = llama_service
        self.semantic_processor = semantic_processor
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Successfully loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {str(e)}")
            raise

        # Schema definitions for validation
        self.entity_schemas = self._load_entity_schemas()
        self.relationship_schemas = self._load_relationship_schemas()

    def process_document(self, file) -> Dict:
        """Process uploaded document and store in knowledge graph with semantic analysis"""
        # Initialize doc_info at the start
        doc_info = {
            'title': file.filename,
            'timestamp': datetime.now().isoformat(),
            'stage': 'extracting',
            'progress': 20
        }

        try:
            # Extract file content
            file_content = self._extract_file_content(file)
            doc_info['content'] = file_content

            # Update progress
            doc_info['stage'] = 'processing'
            doc_info['progress'] = 40

            # Create document node in Neo4j
            self.logger.info("Creating document node in Neo4j...")
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

            # Update progress
            doc_info['stage'] = 'storing'
            doc_info['progress'] = 80

            # Process with LlamaIndex after entity extraction
            self.logger.info("Processing document with LlamaIndex...")
            self.llama_service.process_document(file_content)
            self.logger.info("Document processed successfully with LlamaIndex")

            # Final progress update
            doc_info['stage'] = 'complete'
            doc_info['progress'] = 100

            return doc_info

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            doc_info['stage'] = 'error'
            doc_info['error'] = str(e)
            return doc_info

    def _extract_file_content(self, file) -> str:
        """Extract content from file based on file type"""
        try:
            # Handle both file-like objects and our custom FileWrapper
            if hasattr(file, 'read'):
                content = file.read()
                return content.decode('utf-8') if isinstance(content, bytes) else str(content)
            elif hasattr(file, 'content'):
                return str(file.content)
            else:
                raise ValueError("Invalid file object provided")

        except Exception as e:
            self.logger.error(f"Error extracting file content: {str(e)}")
            raise

    def _extract_entities(self, content: str) -> List[Dict]:
        """Extract domain-specific entities from content"""
        try:
            doc = self.nlp(content)
            entities = []

            # Extract person names as Player entities
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

            # Extract domain entities using rule-based matching
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

            return entities

        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _create_entity_nodes(self, doc_node, entities: List[Dict]) -> None:
        """Create entity nodes and link them to the document"""
        for entity in entities:
            try:
                # Validate entity against schema
                entity_type = entity.get('type')
                if entity_type in self.entity_schemas:
                    schema = self.entity_schemas[entity_type]

                    # Check required fields
                    if all(field in entity for field in schema['required']):
                        # Create the entity and relationship to document
                        self.graph_service.create_entity_node(entity, doc_node)
                    else:
                        self.logger.warning(f"Entity {entity['name']} missing required fields")
                else:
                    # If no schema exists, create anyway but log warning
                    self.logger.warning(f"No schema for entity type: {entity_type}")
                    self.graph_service.create_entity_node(entity, doc_node)
            except Exception as e:
                self.logger.error(f"Error creating entity node: {str(e)}")
                continue

    def _load_entity_schemas(self) -> Dict[str, Dict]:
        """Load entity schemas from configuration"""
        return {
            "Player": {
                "required": ["name"],
                "optional": ["nationality", "achievements", "specialization"],
                "types": {
                    "name": str,
                    "nationality": str,
                    "achievements": str,
                    "specialization": str
                }
            },
            "Skill": {
                "required": ["name"],
                "optional": ["description", "category", "difficulty"],
                "types": {
                    "name": str,
                    "description": str,
                    "category": str,
                    "difficulty": str
                }
            },
            "Drill": {
                "required": ["name"],
                "optional": ["description", "focus_area", "intensity"],
                "types": {
                    "name": str,
                    "description": str,
                    "focus_area": str,
                    "intensity": str
                }
            }
        }

    def _load_relationship_schemas(self) -> Dict[str, Dict]:
        """Load relationship schemas from configuration"""
        return {
            "DEVELOPS": {
                "source": ["Drill"],
                "target": ["Skill"],
                "properties": {
                    "required": [],
                    "optional": ["effectiveness_rating"],
                    "types": {
                        "effectiveness_rating": int
                    }
                }
            },
            "REQUIRES": {
                "source": ["Skill"],
                "target": ["Skill"],
                "properties": {
                    "required": [],
                    "optional": ["strength"],
                    "types": {
                        "strength": int
                    }
                }
            }
        }