import logging
from datetime import datetime
import spacy
import json
import nltk
from typing import Dict, List, Any, Optional

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

    def _load_entity_schemas(self) -> Dict[str, Dict]:
        """Load entity schemas from configuration"""
        return {
            "Skill": {
                "required": ["name"],
                "optional": ["description", "category", "difficulty", "visualRequirements"],
                "types": {
                    "name": str,
                    "description": str,
                    "category": str,
                    "difficulty": str,
                    "visualRequirements": str
                }
            },
            "Drill": {
                "required": ["name"],
                "optional": ["description", "focus_area", "intensity", "duration", 
                           "equipment_needed", "visual_elements", "targets"],
                "types": {
                    "name": str,
                    "description": str,
                    "focus_area": str,
                    "intensity": str,
                    "duration": int,
                    "equipment_needed": str,
                    "visual_elements": str,
                    "targets": str
                }
            },
            "VisualElement": {
                "required": ["name"],
                "optional": ["description", "category"],
                "types": {
                    "name": str,
                    "description": str,
                    "category": str
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
                    "optional": ["primary", "development_phase", "effectiveness_rating"],
                    "types": {
                        "primary": bool,
                        "development_phase": str,
                        "effectiveness_rating": int
                    }
                }
            },
            "REQUIRES": {
                "source": ["Skill"],
                "target": ["Skill"],
                "properties": {
                    "required": [],
                    "optional": ["strength", "transfer_effect"],
                    "types": {
                        "strength": int,
                        "transfer_effect": str
                    }
                }
            }
        }

    def process_document(self, file) -> Dict:
        """Process uploaded document and store in knowledge graph with semantic analysis"""
        try:
            # Create document info
            doc_info = {
                'title': file.filename,
                'timestamp': datetime.now().isoformat()
            }

            # Handle different file types
            file_content = self._extract_file_content(file)
            doc_info['content'] = file_content

            # Extract metadata
            doc_info['metadata'] = self._extract_metadata(file_content)

            # Process with semantic processor if available
            if self.semantic_processor:
                self.logger.info("Processing document with semantic processor...")
                semantic_data = self.semantic_processor.process_document(file_content)
                doc_info['embeddings'] = semantic_data.get('embeddings')
                doc_info['chunks'] = semantic_data.get('chunks')
                self.logger.info(f"Document processed with {len(semantic_data.get('entities', []))} entities extracted")

            # Process with LlamaIndex
            self.logger.info("Processing document with LlamaIndex...")
            self.llama_service.process_document(file_content)
            self.logger.info("Document processed successfully with LlamaIndex")

            # Create document node in Neo4j
            self.logger.info("Creating document node in Neo4j...")
            doc_node = self.graph_service.create_document_node(doc_info)
            self.logger.info("Document node created successfully in Neo4j")

            # Extract and create entity relationships
            self.logger.info("Creating entity relationships...")
            entities = self._extract_entities(file_content)
            self._create_entity_nodes(doc_node, entities)
            self.logger.info(f"Created {len(entities)} entity relationships")

            # Extract and create visual element nodes if present
            visual_elements = self._extract_visual_elements(file_content)
            if visual_elements:
                self._create_visual_element_nodes(doc_node, visual_elements)
                self.logger.info(f"Created {len(visual_elements)} visual element nodes")

            # Extract and create relationships between entities
            relationships = self._extract_relationships(file_content, entities)
            if relationships:
                self._create_relationship_edges(relationships)
                self.logger.info(f"Created {len(relationships)} relationship edges")

            return doc_info

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def _extract_file_content(self, file) -> str:
        """Extract content from file based on file type"""
        if file.filename.endswith('.txt'):
            return file.read().decode('utf-8')
        elif file.filename.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(file)
            return df.to_json(orient='records')
        elif file.filename.endswith(('.pdf', '.doc', '.docx')):
            raise NotImplementedError(
                "Binary file processing is not yet implemented. "
                "Currently supported formats: .txt, .csv"
            )
        else:
            raise ValueError(
                "Unsupported file type. "
                "Currently supported formats: .txt, .csv"
            )

    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from document content"""
        doc = self.nlp(content[:5000])  # Process first 5000 chars for efficiency
        return {
            'word_count': len(content.split()),
            'char_count': len(content),
            'sentence_count': len(list(doc.sents)),
            'created_at': datetime.now().isoformat(),
            'language': doc.lang_
        }

    def _extract_entities(self, content: str) -> List[Dict]:
        """Extract domain-specific entities from content"""
        doc = self.nlp(content)
        entities = []

        # Extract named entities
        for ent in doc.ents:
            entities.append({
                'name': ent.text,
                'type': ent.label_,
                'source': 'spacy_ner'
            })

        # Extract domain entities using rule-based matching
        domain_terms = {
            'Skill': ['setting', 'passing', 'blocking', 'serving', 'attacking', 'digging'],
            'Drill': ['pepper', 'queen of the court', 'mini-game', 'scrimmage', 'target practice'],
            'VisualElement': ['ball tracking', 'peripheral vision', 'trajectory prediction']
        }

        for entity_type, terms in domain_terms.items():
            for term in terms:
                if term.lower() in content.lower():
                    entities.append({
                        'name': term,
                        'type': entity_type,
                        'source': 'domain_terminology'
                    })

        return entities

    def _extract_visual_elements(self, content: str) -> List[Dict]:
        """Extract visual elements specifically from content"""
        visual_terms = [
            'ball tracking', 'peripheral vision', 'trajectory prediction',
            'opponent reading', 'visual focus', 'depth perception',
            'target awareness', 'spatial recognition', 'anticipation',
            'visual scanning', 'court awareness'
        ]

        visual_elements = []
        for term in visual_terms:
            if term.lower() in content.lower():
                visual_elements.append({
                    'name': term,
                    'type': 'VisualElement',
                    'source': 'visual_terminology'
                })

        return visual_elements

    def _extract_relationships(self, content: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities"""
        relationship_patterns = [
            {'source_type': 'Drill', 'relation': 'DEVELOPS', 'target_type': 'Skill'},
            {'source_type': 'Skill', 'relation': 'REQUIRES', 'target_type': 'Skill'},
            {'source_type': 'Drill', 'relation': 'FOCUSES_ON', 'target_type': 'VisualElement'}
        ]

        relationships = []
        sentences = list(self.nlp(content).sents)

        for sentence in sentences:
            sentence_text = sentence.text.lower()

            # Look for entity co-occurrences in the same sentence
            entities_in_sentence = []
            for entity in entities:
                if entity['name'].lower() in sentence_text:
                    entities_in_sentence.append(entity)

            # If we have at least 2 entities in a sentence, check for relationships
            if len(entities_in_sentence) >= 2:
                for i, source_entity in enumerate(entities_in_sentence):
                    for target_entity in entities_in_sentence[i+1:]:
                        # Check if this pair matches any of our relationship patterns
                        for pattern in relationship_patterns:
                            if source_entity['type'] == pattern['source_type'] and \
                               target_entity['type'] == pattern['target_type']:
                                relationships.append({
                                    'source': source_entity['name'],
                                    'source_type': source_entity['type'],
                                    'relation': pattern['relation'],
                                    'target': target_entity['name'],
                                    'target_type': target_entity['type'],
                                    'evidence': sentence_text
                                })

        return relationships

    def _create_entity_nodes(self, doc_node, entities: List[Dict]) -> None:
        """Create entity nodes and link them to the document"""
        for entity in entities:
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

    def _create_visual_element_nodes(self, doc_node, visual_elements: List[Dict]) -> None:
        """Create visual element nodes and link them to the document"""
        for element in visual_elements:
            self.graph_service.create_visual_element_node(element, doc_node)

    def _create_relationship_edges(self, relationships: List[Dict]) -> None:
        """Create relationship edges between entities"""
        for rel in relationships:
            # Validate relationship against schema
            rel_type = rel.get('relation')
            if rel_type in self.relationship_schemas:
                schema = self.relationship_schemas[rel_type]

                # Check if source and target types are valid for this relationship
                if rel['source_type'] in schema['source'] and rel['target_type'] in schema['target']:
                    self.graph_service.create_relationship(
                        source_name=rel['source'],
                        source_type=rel['source_type'],
                        target_name=rel['target'],
                        target_type=rel['target_type'],
                        rel_type=rel_type,
                        properties={'evidence': rel.get('evidence')}
                    )
                else:
                    self.logger.warning(f"Invalid source/target types for relationship: {rel}")
            else:
                self.logger.warning(f"No schema for relationship type: {rel_type}")
                self.graph_service.create_relationship(
                    source_name=rel['source'],
                    source_type=rel['source_type'],
                    target_name=rel['target'],
                    target_type=rel['target_type'],
                    rel_type=rel_type,
                    properties={'evidence': rel.get('evidence')}
                )