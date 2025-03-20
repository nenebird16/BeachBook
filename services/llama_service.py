import os
import logging
from typing import Dict, List, Any, Optional
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from anthropic import Anthropic
from services.semantic_processor import SemanticProcessor

logger = logging.getLogger(__name__)

class LlamaService:
    def __init__(self):
        """Initialize the LlamaService with required components"""
        self.logger = logging.getLogger(__name__)
        self.graph = None
        self.semantic_processor = None

        try:
            self.logger.info("Initializing LlamaService components...")

            # Verify Anthropic API key
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

            # Initialize Anthropic client
            try:
                self.anthropic = Anthropic()
                self.logger.debug("Anthropic client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                raise ValueError(f"Failed to initialize Anthropic client: {str(e)}")

            # Initialize semantic processor (optional)
            try:
                self.semantic_processor = SemanticProcessor()
                self.logger.info("Semantic processor initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize semantic processor: {str(e)}")
                self.logger.warning("Service will continue without semantic processing capabilities")

            # Initialize Neo4j connection (optional)
            try:
                if all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                    from py2neo import Graph, ConnectionProfile
                    from urllib.parse import urlparse

                    uri = urlparse(NEO4J_URI)
                    bolt_uri = f"bolt+s://{uri.netloc}" if uri.scheme == 'neo4j+s' else f"bolt://{uri.netloc}"

                    profile = ConnectionProfile(
                        uri=bolt_uri,
                        user=NEO4J_USER,
                        password=NEO4J_PASSWORD
                    )
                    self.graph = Graph(profile=profile)
                    self.graph.run("RETURN 1 as test").data()
                    self.logger.info("Successfully connected to Neo4j database")
                else:
                    self.logger.warning("Neo4j credentials not configured - graph features will be unavailable")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Neo4j connection: {str(e)}")
                self.logger.warning("Service will continue without graph database capabilities")

        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaService: {str(e)}")
            raise

    def process_query(self, query_text: str) -> Dict[str, Any]:
        """Process a query and generate a response"""
        try:
            self.logger.info(f"Processing query: {query_text}")

            # Get graph context if available
            graph_results = self._get_graph_overview() if self.graph else None

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
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise

    def generate_response(self, query: str, context_info: Optional[str] = None) -> str:
        """Generate a natural language response using Claude"""
        try:
            self.logger.debug("Starting response generation")
            self.logger.debug(f"Query: {query}")
            self.logger.debug(f"Context available: {'Yes' if context_info else 'No'}")

            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            model = "claude-3-5-sonnet-20241022"

            if context_info:
                prompt = f"""Based on the following context from a knowledge graph, help me answer this query: "{query}"

                Context information:
                {context_info}

                Please provide a natural, conversational response that:
                1. Directly answers the query using the context provided
                2. Incorporates relevant information from the context
                3. Highlights key relationships between concepts
                4. Suggests related areas to explore if relevant

                Response:"""
            else:
                # Check if this is a query about graph contents
                is_content_query = any(keyword in query.lower() 
                                    for keyword in ['what', 'tell me about', 'show me', 'list', 'topics'])

                if is_content_query:
                    prompt = "I apologize, but I don't have access to any knowledge graph data at the moment. " \
                            "Please try uploading some documents first or ask a different question."
                else:
                    prompt = f"""I need to respond to this query: "{query}"

                    Since I don't find any matches in the knowledge graph for this query, I should:
                    1. Politely explain that I can only provide information that exists in the knowledge graph
                    2. Suggest that the user ask about specific topics or documents
                    3. Keep the response brief and focused

                    Response:"""

            self.logger.debug("Sending request to Anthropic")
            try:
                response = self.anthropic.messages.create(
                    model=model,
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "assistant",
                            "content": "I am a knowledge graph assistant that only provides information from the connected graph database. I stay focused on available content and politely decline general conversation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                self.logger.debug("Successfully received response from Anthropic")
                return response.content[0].text

            except Exception as e:
                self.logger.error(f"Error calling Anthropic API: {str(e)}")
                self.logger.error(f"Exception type: {type(e)}")
                self.logger.error(f"Exception args: {e.args}")
                raise

        except Exception as e:
            self.logger.error(f"Error generating Claude response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def _get_graph_overview(self) -> Optional[str]:
        """Get an overview of entities and topics in the graph"""
        try:
            if not self.graph:
                return None

            # Get all entity types and their instances
            entity_query = """
            MATCH (e:Entity)
            WITH e.type as type, collect(distinct e.name) as entities
            RETURN type, entities
            ORDER BY size(entities) DESC
            """
            entity_results = self.graph.run(entity_query).data()

            # Get document count and sample titles
            doc_query = """
            MATCH (d:Document)
            RETURN count(d) as doc_count,
                   collect(distinct d.title)[..5] as sample_titles
            """
            doc_results = self.graph.run(doc_query).data()

            if not entity_results and not doc_results[0]['doc_count']:
                return None

            # Format overview
            overview = []

            # Add document information
            if doc_results[0]['doc_count']:
                overview.append(f"Documents: {doc_results[0]['doc_count']} total")
                if doc_results[0]['sample_titles']:
                    overview.append("Sample documents:")
                    for title in doc_results[0]['sample_titles']:
                        overview.append(f"- {title}")
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