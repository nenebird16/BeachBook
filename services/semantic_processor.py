from llama_index.embeddings.openai import OpenAIEmbedding
import logging
from typing import List, Dict, Optional
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import json

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

            self.embed_model = OpenAIEmbedding(
                model_name="text-embedding-3-small",
                dimensions=1536
            )
            self.logger.info("Initialized semantic processing models")
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic processor: {str(e)}")
            raise

    def process_document(self, content: str) -> Dict:
        """Extract semantic information from document"""
        try:
            # Create document chunks
            chunks = self._create_chunks(content)

            # Generate embeddings for chunks
            chunk_embeddings = []
            for chunk in chunks:
                embedding = self.embed_model.get_text_embedding(chunk)
                chunk_embeddings.append({
                    'text': chunk,
                    'embedding': embedding
                })

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
                "chunks": chunks,
                "embeddings": chunk_embeddings
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