import os
import logging
from typing import Dict, List, Any, Optional
from anthropic import Anthropic
from datetime import datetime

class LlamaService:
    def __init__(self, graph_db=None):
        self.logger = logging.getLogger(__name__)
        self.anthropic = Anthropic()
        self.graph_db = graph_db

        if not self.graph_db:
            self.logger.warning("No graph database provided, running in chat-only mode")

    def process_query(self, query_text: str) -> Dict:
        """Process a query using Claude and optionally Neo4j"""
        try:
            # Always start with a basic chat response
            chat_response = self.generate_response(query_text)
            query_analysis = {
                'input_query': query_text,
                'database_state': 'disconnected',
                'query_type': 'content_search',
                'analysis_timestamp': datetime.now().isoformat(),
                'parameters': {'query': query_text}
            }

            # Try to enhance with graph data if available
            if self.graph_db:
                try:
                    self.logger.info("Attempting to query graph database")
                    query_analysis['database_state'] = 'connected'

                    # Define the Cypher queries for different search strategies
                    queries = {
                        'content_search': """
                        MATCH (n)
                        WHERE n.content CONTAINS $query
                        RETURN n.content as content
                        LIMIT 5
                        """,
                        'entity_search': """
                        MATCH (n)-[:RELATES_TO]-(m)
                        WHERE n.content CONTAINS $query
                        RETURN DISTINCT m.content as content
                        LIMIT 3
                        """
                    }

                    # Execute primary content search
                    results = self.graph_db.query(queries['content_search'], {'query': query_text})
                    related_results = []

                    if results:
                        # If we found direct matches, also look for related content
                        self.logger.info(f"Found {len(results)} direct matches in knowledge graph")
                        related_results = self.graph_db.query(queries['entity_search'], {'query': query_text})

                        context = self._prepare_context(results + related_results)
                        if context:
                            chat_response = self.generate_response(query_text, context)

                        query_analysis.update({
                            'found_matches': True,
                            'direct_matches': len(results),
                            'related_matches': len(related_results),
                            'executed_queries': queries
                        })
                    else:
                        self.logger.info("No direct matches found in knowledge graph")
                        query_analysis.update({
                            'found_matches': False,
                            'executed_queries': queries
                        })

                    return {
                        'response': chat_response,
                        'technical_details': {
                            'queries': {
                                'query_analysis': query_analysis,
                                'content_query': queries['content_search'],
                                'entity_query': queries['entity_search'],
                                'parameters': {'query': query_text}
                            },
                            'results': {
                                'direct_matches': results,
                                'related_matches': related_results
                            } if results or related_results else 'No matches found in knowledge graph'
                        }
                    }

                except Exception as e:
                    self.logger.error(f"Error querying graph database: {str(e)}")
                    query_analysis['error'] = str(e)
                    query_analysis['status'] = 'error'

            # Return basic chat response if no graph data
            return {
                'response': chat_response,
                'technical_details': {
                    'queries': {
                        'query_analysis': query_analysis
                    },
                    'results': 'No graph database available'
                }
            }

        except Exception as e:
            self.logger.error(f"Error in process_query: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try again.",
                'error': str(e)
            }

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