import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from anthropic import Anthropic
from services.semantic_processor import SemanticProcessor

class LlamaService:
    def __init__(self, graph_db=None):
        self.logger = logging.getLogger(__name__)
        self.anthropic = Anthropic()
        self.graph_db = graph_db
        self.semantic_processor = SemanticProcessor()

        if not self.graph_db:
            self.logger.warning("No graph database provided, running in chat-only mode")

    def process_query(self, query_text: str) -> Dict:
        """Process a query using Claude and optionally Neo4j"""
        try:
            # Extract entities from query
            query_entities = self.semantic_processor.extract_entities_from_query(query_text)
            self.logger.info(f"Extracted entities from query: {query_entities}")

            # Build search patterns
            search_terms = [entity['text'] for entity in query_entities]
            if not search_terms:  # If no entities found, use the whole query
                search_terms = [query_text.lower()]

            # Create case-insensitive pattern for each term
            search_patterns = [f"(?i).*{term}.*" for term in search_terms]

            # Define base queries for searching
            content_query = """
            MATCH (n)
            WHERE any(pattern IN $search_patterns WHERE 
                  (n:Player AND (n.name =~ pattern OR n.description =~ pattern))
                  OR (n:Skill AND n.name =~ pattern)
                  OR (n:Technique AND n.name =~ pattern)
            )
            WITH n
            OPTIONAL MATCH (n)-[r]->(related)
            RETURN n.name as name, 
                   n.description as description,
                   labels(n) as types,
                   collect(distinct type(r)) as relationships,
                   collect(distinct {type: type(r), target: related.name}) as related_nodes
            LIMIT 5
            """

            entity_query = """
            MATCH (n)-[r]-(m)
            WHERE any(pattern IN $search_patterns WHERE n.name =~ pattern)
            RETURN DISTINCT type(r) as relationship,
                   m.name as related_entity,
                   labels(m) as related_types
            LIMIT 3
            """

            # Generate base chat response
            chat_response = self.generate_response(query_text)

            # Initialize response structure
            response = {
                'response': chat_response,
                'technical_details': {
                    'queries': {
                        'content_query': content_query,
                        'entity_query': entity_query,
                        'parameters': {
                            'search_patterns': search_patterns
                        },
                        'query_analysis': {
                            'input_query': query_text,
                            'query_type': 'volleyball_knowledge_search',
                            'database_state': 'connected' if self.graph_db else 'disconnected',
                            'analysis_timestamp': datetime.now().isoformat(),
                            'found_matches': False,
                            'direct_matches': 0,
                            'related_matches': 0,
                            'entities_found': query_entities
                        }
                    }
                }
            }

            # Try to query graph database if available
            if self.graph_db:
                try:
                    self.logger.info("Attempting to query graph database")

                    # Execute content query
                    results = self.graph_db.query(content_query, {'search_patterns': search_patterns})
                    self.logger.debug(f"Query results: {results}")

                    if results:
                        # If we found direct matches, look for related content
                        self.logger.info(f"Found {len(results)} direct matches in knowledge graph")
                        related_results = self.graph_db.query(entity_query, {'search_patterns': search_patterns})

                        # Prepare context from results
                        context = self._prepare_context(results + related_results)
                        if context:
                            response['response'] = self.generate_response(query_text, context)

                        # Update analysis with match information
                        response['technical_details']['queries']['query_analysis'].update({
                            'found_matches': True,
                            'direct_matches': len(results),
                            'related_matches': len(related_results)
                        })
                    else:
                        self.logger.info("No direct matches found in knowledge graph")

                except Exception as e:
                    self.logger.error(f"Error querying graph database: {str(e)}")
                    response['technical_details']['queries']['query_analysis']['database_state'] = 'error'
                    response['technical_details']['queries']['query_analysis']['error'] = str(e)

            return response

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
            if 'name' in result:
                section = f"Name: {result['name']}"
                if 'description' in result and result['description']:
                    section += f"\nDescription: {result['description']}"
                if 'types' in result:
                    section += f"\nType: {', '.join(result['types'])}"
                if 'relationships' in result and result['relationships']:
                    section += f"\nRelationships: {', '.join(result['relationships'])}"
                if 'related_nodes' in result and result['related_nodes']:
                    related = [f"{r['target']} ({r['type']})" for r in result['related_nodes'] if r['target']]
                    if related:
                        section += f"\nRelated: {', '.join(related)}"
                context_sections.append(section)
            elif 'relationship' in result and 'related_entity' in result:
                section = f"Related: {result['related_entity']}"
                if 'related_types' in result:
                    section += f" (Type: {', '.join(result['related_types'])})"
                context_sections.append(section)

        return "\n\n".join(context_sections) if context_sections else None

    def generate_response(self, query: str, context_info: Optional[str] = None) -> str:
        """Generate a natural language response using Claude"""
        try:
            if context_info:
                prompt = f"""Based on the following context about beach volleyball, answer this query: "{query}"

                Context information:
                {context_info}

                Please provide a natural, conversational response about beach volleyball players, skills, techniques, or training methods.
                """
            else:
                prompt = f"""As a volleyball knowledge assistant, help me with this query: "{query}"

                Please provide a helpful response about volleyball players, skills, techniques, or training methods.
                """

            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {
                        "role": "assistant",
                        "content": "I am a volleyball knowledge assistant that helps with information about players, skills, techniques, and training methods."
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