import os
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from py2neo import Graph, ConnectionProfile
from anthropic import Anthropic

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.anthropic = Anthropic()
        self.graph = None

        # Try to initialize Neo4j connection if credentials are available
        try:
            uri = os.environ.get("NEO4J_URI")
            username = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            if all([uri, username, password]):
                parsed_uri = urlparse(uri)

                # Map Neo4j URI schemes to compatible py2neo schemes
                scheme_mapping = {
                    'neo4j': 'bolt',
                    'neo4j+s': 'bolt+s',
                    'bolt': 'bolt',
                    'bolt+s': 'bolt+s'
                }

                # Get the correct scheme or default to bolt
                original_scheme = parsed_uri.scheme
                scheme = scheme_mapping.get(original_scheme, 'bolt')

                self.logger.info(f"Connecting to Neo4j with scheme: {scheme} (mapped from {original_scheme})")

                profile = ConnectionProfile(
                    scheme=scheme,
                    host=parsed_uri.hostname,
                    port=parsed_uri.port or 7687,
                    secure=scheme.endswith('+s'),
                    user=username,
                    password=password
                )
                self.graph = Graph(profile=profile)
                self.logger.info("Successfully connected to Neo4j database")
            else:
                self.logger.warning("Neo4j credentials not found, running in chat-only mode")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Neo4j: {str(e)}, running in chat-only mode")

    def process_query(self, query_text: str) -> Dict:
        """Process a query using Claude and optionally Neo4j"""
        try:
            # Always start with a basic chat response
            chat_response = self.generate_response(query_text)

            # Try to enhance with graph data if available
            if self.graph:
                try:
                    results = self._execute_knowledge_graph_queries(query_text)
                    if results:
                        context = self._prepare_context(results)
                        if context:
                            chat_response = self.generate_response(query_text, context)

                    return {
                        'response': chat_response,
                        'technical_details': {
                            'queries': {'query': query_text},
                            'results': results
                        }
                    }
                except Exception as e:
                    self.logger.error(f"Error querying graph database: {str(e)}")

            # Return basic chat response if no graph data
            return {
                'response': chat_response,
                'technical_details': {
                    'queries': {'query': query_text},
                    'results': 'No matches found in knowledge graph'
                }
            }

        except Exception as e:
            self.logger.error(f"Error in process_query: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try again.",
                'error': str(e)
            }

    def generate_response(self, query: str, context_info: Optional[str] = None) -> str:
        """Generate a natural language response using Claude"""
        try:
            if context_info:
                prompt = f"""Based on the following context about volleyball, answer this query: "{query}"

                Context information:
                {context_info}

                Please provide a natural, conversational response that directly answers the query.
                """
            else:
                prompt = f"""As a volleyball knowledge assistant, help me with this query: "{query}"

                Please provide a helpful response about volleyball concepts, skills, or training methods.
                """

            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {
                        "role": "assistant",
                        "content": "I am a volleyball knowledge assistant that helps with skills, drills, and practice planning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.content[0].text

        except Exception as e:
            self.logger.error(f"Error generating Claude response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def _execute_knowledge_graph_queries(self, query_text: str) -> List[Dict]:
        """Execute knowledge graph queries based on the query text"""
        if not self.graph:
            return []

        try:
            query = """
            MATCH (n)
            WHERE n.content CONTAINS $query
            RETURN n.content as content
            LIMIT 5
            """
            return self.graph.run(query, query=query_text).data()
        except Exception as e:
            self.logger.error(f"Error executing graph query: {str(e)}")
            return []

    def _prepare_context(self, results: List[Dict]) -> Optional[str]:
        """Prepare context for AI response from results"""
        if not results:
            return None

        context_sections = []
        for result in results:
            if 'content' in result:
                context_sections.append(result['content'])

        return "\n".join(context_sections) if context_sections else None