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
        """Process a query using hybrid search (text + vectors)"""
        try:
            # Extract entities and generate embedding for the query
            query_entities = self.semantic_processor.extract_entities_from_query(query_text)
            query_embedding = self.semantic_processor.generate_embedding(query_text)
            self.logger.info(f"Extracted entities from query: {query_entities}")

            # Build search patterns
            search_terms = [entity['text'] for entity in query_entities]
            if not search_terms:  # If no entities found, use the whole query
                search_terms = [query_text.lower()]

            # Create case-insensitive pattern for each term
            search_patterns = [f"(?i).*{term}.*" for term in search_terms]

            # Vector similarity search query
            vector_query = """
            CALL db.index.vector.queryNodes(
                'document_embeddings',
                5,
                $embedding
            ) YIELD node, score
            WITH node, score
            MATCH (node)-[:CONTAINS]->(e:Entity)
            RETURN node.title as title,
                   node.content as content,
                   collect(distinct e.name) as entities,
                   score as similarity
            ORDER BY similarity DESC
            """

            # Text-based search query
            content_query = """
            MATCH (n)
            WHERE (n:Document AND any(pattern IN $search_patterns 
                  WHERE n.title =~ pattern OR n.content =~ pattern))
               OR (n:Player AND any(pattern IN $search_patterns 
                  WHERE n.name =~ pattern OR n.description =~ pattern))
               OR (n:Skill AND any(pattern IN $search_patterns 
                  WHERE n.name =~ pattern))
               OR (n:Technique AND any(pattern IN $search_patterns 
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

            # Initialize response structure
            response = {
                'response': None,  # Will be set later
                'technical_details': {
                    'queries': {
                        'vector_query': vector_query,
                        'content_query': content_query,
                        'parameters': {
                            'search_patterns': search_patterns,
                            'embedding_dimensions': len(query_embedding) if query_embedding else None
                        },
                        'query_analysis': {
                            'input_query': query_text,
                            'query_type': 'hybrid_search',
                            'database_state': 'connected' if self.graph_db else 'disconnected',
                            'analysis_timestamp': datetime.now().isoformat(),
                            'found_matches': False,
                            'direct_matches': 0,
                            'vector_matches': 0,
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
                    self.logger.info("Executing hybrid search")

                    # Execute vector search if embedding available
                    vector_results = []
                    if query_embedding:
                        vector_results = self.graph_db.query(
                            vector_query, 
                            {'embedding': query_embedding}
                        )
                        self.logger.debug(f"Vector search results: {vector_results}")

                    # Execute text-based search
                    content_results = self.graph_db.query(
                        content_query, 
                        {'search_patterns': search_patterns}
                    )
                    self.logger.debug(f"Text search results: {content_results}")

                    # Combine results
                    all_results = []
                    seen_titles = set()

                    # Add vector results first
                    for result in vector_results:
                        if result.get('title') not in seen_titles:
                            seen_titles.add(result.get('title'))
                            all_results.append(result)

                    # Add text search results
                    for result in content_results:
                        title = result.get('title') or result.get('name')
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            all_results.append(result)

                    if all_results:
                        # Prepare context from combined results
                        context = self._prepare_context(all_results)
                        if context:
                            response['response'] = self.generate_response(query_text, context)

                        # Update analysis
                        response['technical_details']['queries']['query_analysis'].update({
                            'found_matches': True,
                            'direct_matches': len(content_results),
                            'vector_matches': len(vector_results)
                        })

                except Exception as e:
                    self.logger.error(f"Error querying graph database: {str(e)}")
                    response['technical_details']['queries']['query_analysis']['database_state'] = 'error'
                    response['technical_details']['queries']['query_analysis']['error'] = str(e)

            return response

        except Exception as e:
            self.logger.error(f"Error in process_query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
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

            # Handle document results
            if 'title' in result:
                section.append(f"Document: {result['title']}")
                if 'content' in result and result['content']:
                    content_preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                    section.append(f"Content: {content_preview}")
                if 'similarity' in result:
                    section.append(f"Relevance Score: {result['similarity']:.2f}")
                if 'entities' in result:
                    section.append(f"Related concepts: {', '.join(result['entities'])}")

            # Handle player/skill/technique results
            elif 'name' in result:
                section.append(f"Name: {result['name']}")
                if 'description' in result and result['description']:
                    section.append(f"Description: {result['description']}")
                if 'types' in result:
                    section.append(f"Type: {', '.join(result['types'])}")
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

                Please provide a natural, conversational response about volleyball players, skills, techniques, or training methods.
                Focus on the specific information found in the context.
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