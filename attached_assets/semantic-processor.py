from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import spacy
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from datetime import datetime
import numpy as np

class SemanticProcessor:
    """
    Enhanced semantic processor for beach volleyball knowledge graph.
    Extracts domain-specific entities, relationships, and generates embeddings
    for GraphRAG implementation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding()
        # Initialize LLM for zero-shot classification
        self.llm = OpenAI(temperature=0)
        
        # Beach volleyball domain-specific entity patterns
        self.domain_patterns = self._initialize_domain_patterns()
        
        # Visual-motor integration terminology
        self.visual_terms = [
            "ball tracking", "peripheral vision", "trajectory prediction",
            "opponent reading", "visual focus", "depth perception",
            "target awareness", "spatial recognition", "anticipation",
            "visual scanning", "court awareness", "environmental adaptation",
            "visual-motor integration"
        ]
        
        # Training framework terminology
        self.framework_terms = [
            "visual-motor integration", "constraint-led approach",
            "deliberate practice", "periodization", "skill acquisition",
            "motor learning", "perception-action coupling", "decision training"
        ]

    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize domain-specific entity patterns for beach volleyball"""
        return {
            "Skill": [
                "passing", "setting", "hitting", "attacking", "blocking", "serving", 
                "defense", "digging", "jump serve", "float serve", "cut shot", "line shot",
                "pokey", "hand setting", "bump setting", "split blocking", "reading"
            ],
            "Drill": [
                "pepper", "queen of the court", "mini-game", "scrimmage", "wash drill",
                "target practice", "serve receive", "block touch", "transition drill",
                "side-out drill", "defensive drill", "passing progression"
            ],
            "VisualElement": self.visual_terms,
            "Framework": [
                "visual-motor integration", "constraint-led approach",
                "skill acquisition", "motor learning"
            ],
            "Equipment": [
                "volleyball", "court", "net", "antenna", "targets", "cones",
                "agility ladder", "platform", "vision goggles", "resistance bands"
            ]
        }

    def process_document(self, content: str) -> Dict[str, Any]:
        """
        Extract semantic information from document with beach volleyball
        domain-specific entity recognition and embedding generation.
        
        Args:
            content: Document text content
            
        Returns:
            Dictionary with entities, relationships, chunks, and embeddings
        """
        self.logger.info("Processing document for semantic analysis")
        
        # Process with spaCy
        doc = self.nlp(content)
        
        # Extract general entities
        general_entities = self._extract_general_entities(doc)
        
        # Extract domain-specific entities
        domain_entities = self._extract_domain_entities(content)
        
        # Extract visual elements specifically
        visual_elements = self._extract_visual_elements(content)
        
        # Extract relationships between entities
        relationships = self._extract_relationships(content, domain_entities + general_entities)
        
        # Create semantic chunks for embedding
        chunks = self._create_semantic_chunks(content)
        
        # Generate embeddings for chunks
        chunk_embeddings = self._generate_chunk_embeddings(chunks)
        
        # Classify document sections for volleyball domain
        section_classifications = self._classify_document_sections(chunks)
        
        return {
            "general_entities": general_entities,
            "domain_entities": domain_entities,
            "visual_elements": visual_elements,
            "relationships": relationships,
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "section_classifications": section_classifications,
            "processed_at": datetime.now().isoformat()
        }

    def _extract_general_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract general entities using spaCy NER"""
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy_ner"
            })
            
        return entities

    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract beach volleyball domain-specific entities"""
        entities = []
        
        # Lower case content for case-insensitive matching
        content_lower = content.lower()
        
        # Find all domain entities in the content
        for entity_type, patterns in self.domain_patterns.items():
            for pattern in patterns:
                # Use regex to find whole word matches
                for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', content_lower):
                    start_idx = match.start()
                    end_idx = match.end()
                    
                    # Get the original casing from the content
                    original_text = content[start_idx:end_idx]
                    
                    entities.append({
                        "text": original_text,
                        "label": entity_type,
                        "start": start_idx,
                        "end": end_idx,
                        "source": "domain_patterns"
                    })
        
        return entities

    def _extract_visual_elements(self, content: str) -> List[Dict[str, Any]]:
        """Extract visual-motor integration elements specifically"""
        visual_elements = []
        
        # Lower case content for case-insensitive matching
        content_lower = content.lower()
        
        # Find all visual elements in the content
        for term in self.visual_terms:
            # Use regex to find whole word or phrase matches
            for match in re.finditer(r'\b' + re.escape(term) + r'\b', content_lower):
                start_idx = match.start()
                end_idx = match.end()
                
                # Get the original casing from the content
                original_text = content[start_idx:end_idx]
                
                visual_elements.append({
                    "text": original_text,
                    "label": "VisualElement",
                    "start": start_idx,
                    "end": end_idx,
                    "source": "visual_terminology"
                })
        
        return visual_elements

    def _extract_relationships(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities in beach volleyball domain"""
        relationships = []
        
        # Define relationship patterns to look for
        relationship_patterns = [
            {'source_type': 'Drill', 'relation': 'DEVELOPS', 'target_type': 'Skill', 
             'patterns': ['develops', 'improves', 'enhances', 'trains', 'builds']},
            {'source_type': 'Skill', 'relation': 'REQUIRES', 'target_type': 'Skill',
             'patterns': ['requires', 'needs', 'depends on', 'builds on', 'uses']},
            {'source_type': 'Drill', 'relation': 'FOCUSES_ON', 'target_type': 'VisualElement',
             'patterns': ['focuses on', 'emphasizes', 'targets', 'improves', 'addresses']},
            {'source_type': 'Framework', 'relation': 'INFORMS', 'target_type': 'Drill',
             'patterns': ['informs', 'guides', 'shapes', 'structures', 'underpins']}
        ]
        
        # Parse document with spaCy
        doc = self.nlp(content)
        
        # For each sentence, check for co-occurrence of entities and relationship patterns
        for sentence in doc.sents:
            sentence_text = sentence.text.lower()
            
            # Find entities in this sentence
            entities_in_sentence = []
            for entity in entities:
                entity_text = entity['text'].lower()
                if entity_text in sentence_text:
                    entities_in_sentence.append(entity)
            
            # If we have at least 2 entities in the sentence, check for relationships
            if len(entities_in_sentence) >= 2:
                for i, source_entity in enumerate(entities_in_sentence):
                    for target_entity in entities_in_sentence[i+1:]:
                        # Skip if source and target are the same
                        if source_entity['text'].lower() == target_entity['text'].lower():
                            continue
                            
                        # Check for relationship patterns
                        for pattern in relationship_patterns:
                            if source_entity['label'] == pattern['source_type'] and \
                               target_entity['label'] == pattern['target_type']:
                                # Check if any relationship pattern words are in the sentence
                                for rel_pattern in pattern['patterns']:
                                    if rel_pattern in sentence_text:
                                        # Found a relationship match
                                        relationships.append({
                                            'source': source_entity['text'],
                                            'source_type': source_entity['label'],
                                            'relation': pattern['relation'],
                                            'target': target_entity['text'],
                                            'target_type': target_entity['label'],
                                            'evidence': sentence.text,
                                            'confidence': 0.8  # Default confidence score
                                        })
                                        break  # Found a match, no need to check other patterns
        
        return relationships

    def _create_semantic_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into semantic chunks for embedding generation"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            # If adding this sentence would exceed the chunk size, start a new chunk
            if current_length + len(sent.text) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Add sentence to current chunk
            current_chunk.append(sent.text)
            current_length += len(sent.text)
            
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _generate_chunk_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        embeddings = []
        
        for chunk in chunks:
            try:
                # Generate embedding for the chunk
                embedding = self.embed_model.get_text_embedding(chunk)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error generating embedding for chunk: {e}")
                # Add a zero vector as a placeholder
                embeddings.append([0.0] * 1536)  # Standard OpenAI embedding dimensions
        
        return embeddings

    def _classify_document_sections(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Classify document sections for volleyball domain relevance"""
        classifications = []
        
        # Categories for classification
        categories = [
            "skill_description", 
            "drill_explanation", 
            "practice_plan", 
            "visual_training", 
            "framework_explanation",
            "assessment_method",
            "other"
        ]
        
        for i, chunk in enumerate(chunks):
            try:
                # Simple rule-based classification
                category_scores = {}
                
                # Check for skill descriptions
                if any(skill in chunk.lower() for skill in self.domain_patterns["Skill"]):
                    category_scores["skill_description"] = 0.8
                
                # Check for drill explanations
                if any(drill in chunk.lower() for drill in self.domain_patterns["Drill"]):
                    category_scores["drill_explanation"] = 0.8
                
                # Check for practice plans
                if "practice plan" in chunk.lower() or "session" in chunk.lower():
                    category_scores["practice_plan"] = 0.8
                
                # Check for visual training content
                if any(term in chunk.lower() for term in self.visual_terms):
                    category_scores["visual_training"] = 0.8
                
                # Check for framework explanations
                if any(term in chunk.lower() for term in self.framework_terms):
                    category_scores["framework_explanation"] = 0.8
                
                # Check for assessment methods
                if "assessment" in chunk.lower() or "evaluation" in chunk.lower():
                    category_scores["assessment_method"] = 0.8
                
                # If no categories matched, mark as other
                if not category_scores:
                    category_scores["other"] = 0.8
                
                # Find the highest scoring category
                best_category = max(category_scores.items(), key=lambda x: x[1])
                
                classifications.append({
                    "chunk_index": i,
                    "category": best_category[0],
                    "confidence": best_category[1],
                    "categories": category_scores
                })
                
            except Exception as e:
                self.logger.error(f"Error classifying chunk {i}: {e}")
                classifications.append({
                    "chunk_index": i,
                    "category": "other",
                    "confidence": 0.5,
                    "categories": {"other": 0.5}
                })
        
        return classifications

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        try:
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(query)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            # Return a zero vector
            return [0.0] * 1536  # Standard OpenAI embedding dimensions

    def find_similar_chunks(self, query_embedding: List[float], chunk_embeddings: List[List[float]], 
                           chunks: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Find chunks similar to the query using embeddings"""
        results = []
        
        # Calculate cosine similarity between query and all chunks
        similarities = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k results
        for i, similarity in similarities[:top_k]:
            results.append({
                "chunk_index": i,
                "chunk_text": chunks[i],
                "similarity": similarity
            })
        
        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        if np.linalg.norm(a) * np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def extract_entities_from_query(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from a search query for graph traversal"""
        entities = []
        
        # Process with spaCy
        doc = self.nlp(query)
        
        # Extract general named entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "source": "spacy_ner"
            })
        
        # Extract domain-specific entities
        for entity_type, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if pattern.lower() in query.lower():
                    entities.append({
                        "text": pattern,
                        "label": entity_type,
                        "source": "domain_patterns"
                    })
        
        return entities
