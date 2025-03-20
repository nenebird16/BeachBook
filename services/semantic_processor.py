
import logging
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)

class SemanticProcessor:
    def __init__(self):
        """Initialize the semantic processor with spaCy model"""
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize spaCy model
            self.logger.info("Initializing spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_md")
            except OSError:
                self.logger.warning("SpaCy model not found, downloading now...")
                spacy.cli.download("en_core_web_md")
                self.nlp = spacy.load("en_core_web_md")
            
            self.logger.info("Successfully initialized semantic processing")

        except Exception as e:
            self.logger.error(f"Failed to initialize semantic processor: {str(e)}")
            raise

    def get_text_embedding(self, text: str) -> list:
        """Get text embedding using spaCy"""
        try:
            doc = self.nlp(text)
            return doc.vector.tolist()
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {str(e)}")
            raise

    def process_document(self, content: str) -> dict:
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
                embedding = self.get_text_embedding(chunk)
                embeddings.append({
                    "text": chunk,
                    "embedding": embedding
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

    def analyze_query(self, query: str) -> dict:
        """Analyze query for semantic search"""
        try:
            self.logger.debug(f"Analyzing query: {query}")

            # Generate query embedding
            query_embedding = self.get_text_embedding(query)
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
                'embedding': query_embedding,
                'entities': query_entities,
                'focus': focus
            }
            self.logger.debug("Query analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            raise

    def _create_chunks(self, text: str, chunk_size: int = 512) -> list:
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
