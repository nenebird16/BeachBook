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
        try:
            # Create document info
            doc_info = {
                'title': file.filename,
                'timestamp': datetime.now().isoformat()
            }

            # Extract content from file
            file_content = self._extract_file_content(file)
            doc_info['content'] = file_content

            # Extract metadata
            doc_info['metadata'] = self._extract_metadata(file_content)

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

            return doc_info

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def _extract_file_content(self, file) -> str:
        """Extract content from file based on file type"""
        try:
            if file.filename.endswith('.txt'):
                content = file.read()
                text_content = content.decode('utf-8') if isinstance(content, bytes) else str(content)
                self.logger.debug(f"Extracted text content: {text_content[:100]}...")  # Log first 100 chars
                return text_content
            elif file.filename.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(file)
                return df.to_string(index=False)  # Convert to plain text format
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
        except Exception as e:
            self.logger.error(f"Error extracting file content: {str(e)}")
            raise

    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from document content"""
        try:
            doc = self.nlp(content[:5000])  # Process first 5000 chars for efficiency
            return {
                'word_count': len(content.split()),
                'char_count': len(content),
                'sentence_count': len(list(doc.sents)),
                'created_at': datetime.now().isoformat(),
                'language': doc.lang_
            }
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}

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

            # Add relationships between players mentioned together
            player_entities = [e for e in entities if e['type'] == 'Player']
            for i, player1 in enumerate(player_entities):
                for player2 in player_entities[i+1:]:
                    if abs(content.find(player1['name']) - content.find(player2['name'])) < 100:
                        # Players mentioned close together likely have a relationship
                        entities.append({
                            'name': f"{player1['name']} and {player2['name']}",
                            'type': 'Partnership',
                            'source': 'relationship_inference'
                        })

            return entities

        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _load_entity_schemas(self) -> Dict[str, Dict]:
        """Load entity schemas from configuration"""
        return {
            "Player": {
                "required": ["name"],
                "optional": ["nationality", "achievements", "specialization", 
                           "playing_style", "team_partner", "visual_strengths"],
                "types": {
                    "name": str,
                    "nationality": str,
                    "achievements": str,
                    "specialization": str,
                    "playing_style": str,
                    "team_partner": str,
                    "visual_strengths": str
                }
            },
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