import os
import logging
import time
from typing import Dict, List, Any, Optional
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from anthropic import Anthropic
from services.semantic_processor import SemanticProcessor

logger = logging.getLogger(__name__)

class LlamaService:
    def __init__(self):
        """Initialize the LlamaService with required components"""
        self.logger = logging.getLogger(__name__)
        self._anthropic = None
        self._graph = None
        self._semantic_processor = None

        # Initialize only the core Anthropic client
        self._init_anthropic()

        # Log initialization status
        self.logger.info("LlamaService initialization complete. Status:")
        self.logger.info(f"- Anthropic client: {'Available' if self._anthropic else 'Unavailable'}")
        self.logger.info("Optional components will be initialized on first use")

    def _init_anthropic(self):
        """Initialize the Anthropic client and semantic processor"""
        try:
            start_time = time.time()
            api_key = os.environ.get('ANTHROPIC_API_KEY')

            if not api_key:
                self.logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                return

            try:
                self._anthropic = Anthropic()
                self._semantic_processor = SemanticProcessor()
                init_time = time.time() - start_time
                self.logger.info(f"Anthropic client and semantic processor initialized successfully in {init_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to initialize services: {str(e)}")
                self._anthropic = None
                self._semantic_processor = None

        except Exception as e:
            self.logger.error(f"Error during service initialization: {str(e)}", exc_info=True)
            self._anthropic = None
            self._semantic_processor = None

    @property
    def anthropic(self):
        """Lazy-loaded Anthropic client"""
        return self._anthropic

    @property
    def graph(self):
        """Lazy-loaded Neo4j graph connection"""
        if self._graph is None:
            try:
                if all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                    start_time = time.time()
                    from py2neo import Graph, ConnectionProfile
                    from urllib.parse import urlparse

                    uri = urlparse(NEO4J_URI)
                    bolt_uri = f"bolt+s://{uri.netloc}" if uri.scheme == 'neo4j+s' else f"bolt://{uri.netloc}"

                    profile = ConnectionProfile(
                        uri=bolt_uri,
                        user=NEO4J_USER,
                        password=NEO4J_PASSWORD
                    )
                    self._graph = Graph(profile=profile)
                    self._graph.run("RETURN 1 as test").data()
                    init_time = time.time() - start_time
                    self.logger.info(f"Neo4j connection established in {init_time:.2f} seconds")
                else:
                    self.logger.warning("Neo4j credentials not configured - graph features will be unavailable")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Neo4j connection: {str(e)}")
                self._graph = None
        return self._graph

    def process_query(self, query_text: str) -> Dict[str, Any]:
        """Process a query and generate a response"""
        try:
            if not self.anthropic:
                return {
                    'response': "I apologize, but the knowledge service is currently unavailable. Please try again later.",
                    'technical_details': {
                        'queries': {}
                    }
                }

            self.logger.info(f"Processing query: {query_text}")

            # Get graph context if available (lazy-loaded)
            graph_results = self._get_graph_overview(query_text) if self.graph else None

            # Generate response using Claude
            response = self.generate_response(query_text, graph_results)

            return {
                'response': response,
                'technical_details': {
                    'queries': {
                        'graph_context': graph_results
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                'response': "I encountered an error while processing your request. Please try again.",
                'technical_details': {
                    'queries': {}
                }
            }

    def generate_response(self, query: str, context_info: Optional[str] = None) -> str:
        """Generate a natural language response using Claude"""
        try:
            if not self.anthropic:
                return "The knowledge service is currently unavailable. Please try again later."

            self.logger.debug("Starting response generation")
            self.logger.debug(f"Query: {query}")
            self.logger.debug(f"Context available: {'Yes' if context_info else 'No'}")

            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            model = "claude-3-5-sonnet-20241022"

            system_message = "I am a knowledge graph assistant that only provides information from the connected graph database. I stay focused on available content and politely decline general conversation."

            if context_info:
                user_message = f"""Based on the following context from a knowledge graph, help me answer this query: "{query}"

Context information:
{context_info}

Please provide a natural, conversational response that:
1. Directly answers the query using the context provided
2. Incorporates relevant information from the context
3. Highlights key relationships between concepts
4. Suggests related areas to explore if relevant"""
            else:
                is_content_query = any(keyword in query.lower() for keyword in 
                    ['what', 'tell me about', 'show me', 'list', 'topics'])

                if is_content_query:
                    user_message = ("I apologize, but I don't have access to any knowledge graph data at the moment. "
                                  "Please try uploading some documents first or ask a different question.")
                else:
                    user_message = f"""I need to respond to this query: "{query}"

Since I don't find any matches in the knowledge graph for this query, I should:
1. Politely explain that I can only provide information that exists in the knowledge graph
2. Suggest that the user ask about specific topics or documents
3. Keep the response brief and focused"""

            try:
                self.logger.debug("Sending request to Anthropic")
                response = self.anthropic.messages.create(
                    model=model,
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[
                        {"role": "assistant", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                )
                self.logger.debug("Successfully received response from Anthropic")
                return response.content[0].text

            except Exception as e:
                self.logger.error(f"Error calling Anthropic API: {str(e)}", exc_info=True)
                self.logger.error(f"Exception type: {type(e)}")
                self.logger.error(f"Exception args: {e.args}")
                raise

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def _get_graph_overview(self, query_text: str) -> Optional[str]:
        """Get an overview of entities and topics in the graph"""
        try:
            if not self.graph:
                return None

            keyword = query_text.lower()

            # Get all entity types and their instances
            entity_query = """
            MATCH (e:Entity)
            WITH e.type as type, collect(distinct e.name) as entities
            RETURN type, entities
            ORDER BY size(entities) DESC
            """
            entity_results = self.graph.run(entity_query).data()

            # Enhanced hybrid retrieval combining semantic and graph structure
            doc_query = f"""
            MATCH (d:Document)-[r:CONTAINS]->(e:Entity)
            WHERE d.content IS NOT NULL
            WITH d, 
                 toLower(d.title) as cleaned_title, 
                 toLower(d.content) as cleaned_content,
                 d.embedding as doc_embedding,
                 $embedding as query_embedding
            WITH d, cleaned_title, cleaned_content,
                 CASE 
                    WHEN doc_embedding IS NOT NULL
                    THEN reduce(dot = 0.0, i IN range(0, size(doc_embedding)-1) | 
                         dot + doc_embedding[i] * query_embedding[i]) /
                         (sqrt(reduce(norm = 0.0, i IN range(0, size(doc_embedding)-1) | 
                         norm + doc_embedding[i] * doc_embedding[i])) *
                         sqrt(reduce(norm = 0.0, i IN range(0, size(query_embedding)-1) | 
                         norm + query_embedding[i] * query_embedding[i])))
                    ELSE 0.0
                 END as embedding_score
            WHERE embedding_score > 0.3 OR cleaned_title CONTAINS $keyword OR cleaned_content CONTAINS $keyword
            RETURN d {.title, .content}, embedding_score
            ORDER BY embedding_score DESC
            LIMIT 5
            """
            doc_results = self.graph.run(doc_query, embedding=self._semantic_processor.get_text_embedding(query_text), keyword=keyword).data()


            if not entity_results and not doc_results:
                return None

            # Format overview
            overview = []

            # Add document information
            if doc_results:
                overview.append(f"Documents:")
                for result in doc_results:
                    overview.append(f"- {result['d']['title']}")
                overview.append("")

            # Add entity information
            if entity_results:
                overview.append("Topics and concepts found:")
                for result in entity_results:
                    if result['type'] and result['entities']:  # Only show non-empty categories
                        entity_type = result['type']
                        entities = result['entities'][:5]  # Limit to 5 examples
                        overview.append(f"- {entity_type.title()}: {', '.join(entities)}")
                overview.append("")

            return "\n".join(overview)

        except Exception as e:
            self.logger.error(f"Error getting graph overview: {str(e)}")
            return None