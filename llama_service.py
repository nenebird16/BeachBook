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
        """Process a query using Neo4j and Claude"""
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

            # Define main search query
            content_query = """
            MATCH (n)
            WHERE (n:Document AND any(pattern IN $search_patterns 
                  WHERE n.title =~ pattern OR n.content =~ pattern))
               OR (n:Player AND any(pattern IN $search_patterns 
                  WHERE n.name =~ pattern OR n.description =~ pattern))
               OR (n:Skill AND any(pattern IN $search_patterns 
                  WHERE n.name =~ pattern))
            WITH n
            OPTIONAL MATCH (n)-[r]->(related)
            RETURN n.title as title,
                   n.content as content,
                   n.name as name,
                   n.description as description,
                   labels(n) as types,
                   collect(distinct type(r)) as relationships,
                   collect(distinct {type: type(r), target: related.name}) as related_nodes
            LIMIT 5
            """

            # Entity relationship query
            entity_query = """
            MATCH (n)-[r]-(m)
            WHERE any(pattern IN $search_patterns WHERE n.name =~ pattern)
            RETURN DISTINCT type(r) as relationship,
                   m.name as related_entity,
                   labels(m) as related_types
            LIMIT 3
            """

            # Initialize response structure
            response = {
                'response': None,
                'technical_details': {
                    'queries': {
                        'content_query': content_query,
                        'entity_query': entity_query,
                        'parameters': {
                            'search_patterns': search_patterns
                        },
                        'query_analysis': {
                            'input_query': query_text,
                            'query_type': 'knowledge_search',
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

            # Generate base chat response
            response['response'] = self.generate_response(query_text)

            # Try to query graph database if available
            if self.graph_db:
                try:
                    self.logger.info("Executing knowledge graph query")

                    # Execute main content query
                    results = self.graph_db.query(content_query, {'search_patterns': search_patterns})
                    self.logger.debug(f"Content query results: {results}")

                    if results:
                        self.logger.info(f"Found {len(results)} matches in knowledge graph")

                        # Look for related content
                        related_results = self.graph_db.query(entity_query, {'search_patterns': search_patterns})
                        self.logger.debug(f"Related query results: {related_results}")

                        # Prepare context from results
                        context = self._prepare_context(results + (related_results or []))
                        if context:
                            response['response'] = self.generate_response(query_text, context)

                        # Update analysis
                        response['technical_details']['queries']['query_analysis'].update({
                            'found_matches': True,
                            'direct_matches': len(results),
                            'related_matches': len(related_results) if related_results else 0
                        })

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
            section = []

            # Handle Document results
            if 'title' in result:
                section.append(f"Document: {result['title']}")
                if 'content' in result and result['content']:
                    content_preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                    section.append(f"Content: {content_preview}")

            # Handle Player/Skill results
            if 'name' in result:
                section.append(f"Name: {result['name']}")
                if 'description' in result and result['description']:
                    section.append(f"Description: {result['description']}")
                if 'types' in result:
                    section.append(f"Type: {', '.join(result['types'])}")

            # Handle relationships
            if 'relationships' in result and result['relationships']:
                section.append(f"Relationships: {', '.join(result['relationships'])}")
            if 'related_nodes' in result and result['related_nodes']:
                related = [f"{r['target']} ({r['type']})" for r in result['related_nodes'] if r['target']]
                if related:
                    section.append(f"Related: {', '.join(related)}")

            if section:
                context_sections.append("\n".join(section))

        return "\n\n".join(context_sections)

    def generate_response(self, query: str, context_info: Optional[str] = None) -> str:
        """Generate a natural language response using Claude"""
        try:
            if context_info:
                prompt = f"""Based on the following context about volleyball, help me answer this query: "{query}"

                Context information:
                {context_info}

                Please provide a natural, conversational response that includes relevant information from the context.
                Focus on answering the specific query while highlighting key relationships between concepts.
                """
            else:
                prompt = f"""As a volleyball knowledge assistant, help me with this query: "{query}"

                Please provide a helpful response about volleyball players, skills, techniques, or training methods.
                If no specific information is found, suggest exploring related topics.
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