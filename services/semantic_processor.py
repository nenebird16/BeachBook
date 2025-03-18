
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import spacy
from typing import List, Dict
import logging

class SemanticProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_sm")
        self.embed_model = OpenAIEmbedding()
        self.llm = OpenAI(temperature=0)

    def process_document(self, content: str) -> Dict:
        """Extract semantic information from document"""
        doc = self.nlp(content)
        
        # Extract entities and relationships
        entities = []
        relationships = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        # Create document chunks
        chunks = self._create_chunks(content)
        
        # Generate embeddings for chunks
        embeddings = self.embed_model.get_text_embedding(
            text="\n".join(chunks)
        )
        
        return {
            "entities": entities,
            "relationships": relationships,
            "chunks": chunks,
            "embeddings": embeddings
        }

    def _create_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into semantic chunks"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            if current_length + len(sent.text) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sent.text)
            current_length += len(sent.text)
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
