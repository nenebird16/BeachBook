import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
from py2neo import Graph, ConnectionProfile, Node, Relationship
from anthropic import Anthropic
import json
from datetime import datetime

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize Anthropic client for Claude
        self.anthropic = Anthropic()

        try:
            uri = os.environ.get("NEO4J_URI")
            username = os.environ.get("NEO4J_USER")
            password = os.environ.get("NEO4J_PASSWORD")

            if not all([uri, username, password]):
                self.logger.error("Neo4j credentials not properly configured")
                raise ValueError("Neo4j credentials not properly configured")

            # Parse URI for AuraDB
            parsed_uri = urlparse(uri)
            self.logger.debug(f"Original URI scheme: {parsed_uri.scheme}")
            self.logger.debug(f"Original URI netloc: {parsed_uri.netloc}")

            # Initialize direct Neo4j connection
            profile = ConnectionProfile(
                scheme="bolt+s" if parsed_uri.scheme == 'neo4j+s' else parsed_uri.scheme,
                host=parsed_uri.netloc,
                port=7687,
                secure=True if parsed_uri.scheme == 'neo4j+s' else False,
                user=username,
                password=password
            )
            self.graph = Graph(profile=profile)
            self.logger.info("Successfully connected to Neo4j database")

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j connections: {str(e)}")
            raise

    def process_query(self, query_text: str) -> Dict:
        """Process a query using Neo4j and Anthropic"""
        try:
            self.logger.info(f"Processing query: {query_text}")
            
            # Execute knowledge graph queries
            results = self._execute_knowledge_graph_queries(query_text)
            
            # Prepare context for AI response
            context_info = self._prepare_context(results)
            
            # Generate AI response with or without context
            ai_response = self.generate_response(query_text, context_info)

            return {
                'chat_response': ai_response,
                'queries': {
                    'query_analysis': {'query': query_text},
                    'results': results
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                'chat_response': 'I apologize, but I encountered an error while processing your query. Please try again.',
                'error': str(e)
            }

    def _execute_knowledge_graph_queries(self, query_text: str) -> List[Dict]:
        """Execute knowledge graph queries based on the query text"""
        try:
            # Simple query to match content
            query = """
            MATCH (n)
            WHERE n.content CONTAINS $query
            RETURN n.content as content
            LIMIT 5
            """
            return self.graph.run(query, query=query_text).data()
            
        except Exception as e:
            self.logger.error(f"Error executing knowledge graph queries: {str(e)}")
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
                prompt = f"""As a volleyball knowledge assistant, I need to respond to this query: "{query}"

                Since I don't find any specific matches in the knowledge base for this query, I should:
                1. Politely explain that I can only provide information that exists in the volleyball knowledge base
                2. Suggest asking about volleyball skills, drills, or practice plans
                """

            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {
                        "role": "assistant",
                        "content": "I am a volleyball knowledge assistant that provides information from our connected database focusing on skills, drills, and practice planning."
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
