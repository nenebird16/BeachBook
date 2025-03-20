
import logging
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)

class SemanticProcessor:
    def __init__(self):
        """Initialize the semantic processor with transformer models"""
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize transformer models
            self.logger.info("Initializing transformer models...")
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModel.from_pretrained('distilbert-base-uncased')
            self.logger.info("Successfully initialized semantic processing")
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic processor: {str(e)}")
            raise

    def get_text_embedding(self, text: str) -> list:
        """Get text embedding using transformers"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings[0].numpy().tolist()
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {str(e)}")
            raise

    def process_document(self, content: str) -> dict:
        """Extract semantic information from document"""
        try:
            # Create document chunks
            chunks = self._create_chunks(content)
            
            # Process with transformers
            inputs = self.tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract entities (using basic token analysis)
            tokens = self.tokenizer.tokenize(content)
            entities = []
            current_entity = []
            
            for token in tokens:
                if token.startswith('##'):
                    if current_entity:
                        current_entity.append(token[2:])
                else:
                    if current_entity:
                        entities.append({
                            "text": ''.join(current_entity),
                            "label": "ENTITY",
                            "start": len(''.join(tokens[:tokens.index(current_entity[0])])),
                            "end": len(''.join(tokens[:tokens.index(token)]))
                        })
                        current_entity = []
                    if token[0].isupper():
                        current_entity.append(token)

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

            # Extract query tokens
            tokens = self.tokenizer.tokenize(query)
            query_entities = [{'text': token, 'label': 'TOKEN'} 
                            for token in tokens if not token.startswith('##')]

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
