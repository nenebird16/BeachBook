import logging
from typing import List, Dict, Optional
import spacy
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class SemanticProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.info("Downloading spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            # Initialize sentence transformer for embeddings
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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

            self.logger.info("Initialized semantic processing models")
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

            return {
                "entities": entities,
                "relationships": relationships,
                "chunks": chunks
            }

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def analyze_query(self, query: str) -> Dict:
        """Analyze query for semantic search"""
        try:
            # Generate query embedding
            query_embedding = self.embed_model.get_text_embedding(query)

            # Extract query entities and intent
            doc = self.nlp(query)
            query_entities = [{'text': ent.text, 'label': ent.label_}
                            for ent in doc.ents]

            # Identify main focus of query
            root = next(token for token in doc if token.head == token)
            focus = {
                'root_verb': root.text if root.pos_ == 'VERB' else None,
                'main_noun': next((token.text for token in doc
                                 if token.pos_ == 'NOUN'), None)
            }

            return {
                'embedding': query_embedding,
                'entities': query_entities,
                'focus': focus
            }

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