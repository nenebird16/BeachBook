import logging
from typing import List, Dict, Optional
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Download required NLTK data
nltk.download('punkt', quiet=True)

class SemanticProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize spaCy model
            self.logger.info("Initializing spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Successfully loaded spaCy model")
            except OSError:
                self.logger.info("Downloading spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            # Initialize sentence transformer for embeddings
            self.logger.info("Initializing SentenceTransformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            if not hasattr(self.model, 'encode'):
                raise ValueError("SentenceTransformer model not properly initialized")
            # Test encoding to verify model
            test_encoding = self.model.encode("Test sentence")
            if test_encoding is None or len(test_encoding) == 0:
                raise ValueError("SentenceTransformer model failed to generate embeddings")
            self.logger.info("Successfully loaded and verified SentenceTransformer model")

            # Initialize domain patterns
            self.domain_patterns = {
                "Skill": [
                    "passing", "setting", "hitting", "attacking", "blocking", "serving", 
                    "defense", "digging", "jump serve", "float serve", "cut shot"
                ],
                "Drill": [
                    "pepper", "queen of the court", "mini-game", "scrimmage",
                    "target practice", "serve receive", "block touch"
                ],
                "VisualElement": [
                    "ball tracking", "peripheral vision", "trajectory prediction",
                    "depth perception", "pattern recognition", "visual focus"
                ]
            }

            self.logger.info("Initialized semantic processing models successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic processor: {str(e)}")
            raise

    def extract_entities_from_query(self, query: str) -> List[Dict[str, str]]:
        """Extract entities from a search query"""
        try:
            entities = []
            doc = self.nlp(query)

            # Extract named entities
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'type': 'named_entity'
                })

            # Look for domain-specific keywords
            domain_keywords = {
                'player': ['player', 'athlete', 'professional'],
                'skill': ['serve', 'block', 'spike', 'dig', 'set'],
                'technique': ['approach', 'footwork', 'positioning'],
                'drill': ['drill', 'exercise', 'practice', 'training']
            }

            query_lower = query.lower()
            for category, keywords in domain_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        entities.append({
                            'text': keyword,
                            'label': category.upper(),
                            'type': 'domain_keyword'
                        })

            self.logger.debug(f"Extracted entities from query: {entities}")
            return entities

        except Exception as e:
            self.logger.error(f"Error extracting entities from query: {str(e)}")
            return []  # Return empty list on error, let calling code handle this gracefully

    def process_document(self, content: str) -> Dict:
        """Extract semantic information from document"""
        try:
            # Create document chunks
            chunks = self._create_chunks(content)

            # Extract entities and relationships
            doc = self.nlp(content)
            entities = []
            relationships = []

            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            # Extract basic relationships between entities
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    relationships.append({
                        "subject": token.head.text,
                        "predicate": token.dep_,
                        "object": token.text
                    })

            # Generate embeddings for chunks
            embeddings = []
            for chunk in chunks:
                embedding = self.model.encode(chunk)
                embeddings.append({
                    "text": chunk,
                    "embedding": embedding.tolist()
                })

            return {
                "entities": entities,
                "relationships": relationships,
                "chunks": chunks,
                "embeddings": embeddings
            }

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def analyze_query(self, query: str) -> Dict:
        """Analyze query for semantic search"""
        try:
            self.logger.debug(f"Analyzing query: {query}")

            # Generate query embedding
            query_embedding = self.model.encode(query)
            self.logger.debug("Generated query embedding successfully")

            # Extract query entities and intent
            doc = self.nlp(query)
            query_entities = [{'text': ent.text, 'label': ent.label_}
                            for ent in doc.ents]
            self.logger.debug(f"Extracted entities: {query_entities}")

            # Identify main focus of query
            root = next(token for token in doc if token.head == token)
            focus = {
                'root_verb': root.text if root.pos_ == 'VERB' else None,
                'main_noun': next((token.text for token in doc
                                 if token.pos_ == 'NOUN'), None)
            }
            self.logger.debug(f"Query focus: {focus}")

            result = {
                'embedding': query_embedding.tolist(),
                'entities': query_entities,
                'focus': focus
            }
            self.logger.debug("Query analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            raise

    def _create_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into semantic chunks"""
        try:
            # First split into sentences
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_length = 0

            for sent in sentences:
                # +1 for space separator
                if current_length + len(sent) + 1 > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_length = len(sent)
                else:
                    current_chunk.append(sent)
                    current_length += len(sent) + 1

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks
        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            return [text]  # Return the full text as a single chunk if chunking fails