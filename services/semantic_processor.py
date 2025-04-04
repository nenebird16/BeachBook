import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any
import spacy

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)

class SemanticProcessor:
    def __init__(self):
        """Initialize the semantic processor with sentence transformers and spaCy"""
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize sentence transformer model
            self.logger.info("Initializing sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Successfully initialized sentence transformer")
            # Initialize spaCy NLP model
            self.logger.info("Initializing spaCy NLP model...")
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Successfully initialized spaCy NLP model")
            self.logger.info("Successfully initialized semantic processing")
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic processor: {str(e)}")
            raise

    def get_text_embedding(self, text: str) -> list:
        """Get text embedding using sentence-transformers"""
        try:
            embeddings = self.model.encode(text, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {str(e)}")
            raise

    def process_document(self, content: str) -> dict:
        """Extract semantic information from document"""
        try:
            # Create document chunks
            chunks = self._create_chunks(content)

            # Generate embeddings for chunks
            embeddings = []
            for chunk in chunks:
                embedding = self.get_text_embedding(chunk)
                embeddings.append({
                    "text": chunk,
                    "embedding": embedding
                })

            # Extract entities using enhanced NLP techniques
            entities = self._extract_entities(content)

            return {
                "entities": entities,
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

            # Basic query analysis
            query_entities = self._extract_entities(query)

            result = {
                'embedding': query_embedding,
                'entities': query_entities
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

    def _extract_entities(self, text: str) -> list:
        """Extract entities using enhanced NLP techniques"""
        entities = []
        doc = self.nlp(text)

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "context": text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
            })

        return entities