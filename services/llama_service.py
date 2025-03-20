import os
import logging
import time
from typing import Dict, List, Any, Optional
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from anthropic import Anthropic
from openai import OpenAI
from services.semantic_processor import SemanticProcessor

logger = logging.getLogger(__name__)

class LlamaService:
    def __init__(self):
        """Initialize the LlamaService with required components"""
        self.logger = logging.getLogger(__name__)
        self._anthropic = None
        self._openai = None
        self._graph = None
        self._semantic_processor = None

        # Try to initialize LLM clients
        self._init_llm_clients()

        # Log initialization status
        self.logger.info("LlamaService initialization complete. Status:")
        self.logger.info(f"- Anthropic client: {'Available' if self._anthropic else 'Unavailable'}")
        self.logger.info(f"- OpenAI client: {'Available' if self._openai else 'Unavailable'}")
        self.logger.info("Optional components will be initialized on first use")

    def _init_llm_clients(self):
        """Initialize available LLM clients"""
        try:
            start_time = time.time()
            
            # Try Anthropic first
            anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
            if anthropic_key:
                try:
                    self._anthropic = Anthropic()
                    self.logger.info("Anthropic client initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Anthropic: {str(e)}")
                    self._anthropic = None
            else:
                self.logger.warning("ANTHROPIC_API_KEY not found")

            # Try OpenAI if Anthropic is not available
            if not self._anthropic:
                openai_key = os.environ.get('OPENAI_API_KEY')
                if openai_key:
                    try:
                        self._openai = OpenAI()
                        self.logger.info("OpenAI client initialized successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize OpenAI: {str(e)}")
                        self._openai = None
                else:
                    self.logger.warning("OPENAI_API_KEY not found")

            # Initialize semantic processor if any LLM client is available
            if self._anthropic or self._openai:
                self._semantic_processor = SemanticProcessor()
                init_time = time.time() - start_time
                self.logger.info(f"Services initialized in {init_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error during service initialization: {str(e)}", exc_info=True)
            self._anthropic = None
            self._openai = None
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
            if not (self._anthropic or self._openai):
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
        """Generate a natural language response using available LLM"""
        try:
            if not self._anthropic and not self._openai:
                return "The knowledge service is currently unavailable. Please try again later."

            self.logger.debug("Starting response generation")
            self.logger.debug(f"Query: {query}")
            self.logger.debug(f"Context available: {'Yes' if context_info else 'No'}")

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
                if self._anthropic:
                    self.logger.debug("Using Anthropic for response generation")
                    response = self._anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        temperature=0.7,
                        messages=[
                            {"role": "assistant", "content": system_message},
                            {"role": "user", "content": user_message}
                        ]
                    )
                    return response.content[0].text
                else:
                    self.logger.debug("Using OpenAI for response generation")
                    response = self._openai.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        max_tokens=1000,
                        temperature=0.7,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ]
                    )
                    return response.choices[0].message.content

            except Exception as e:
                self.logger.error(f"Error calling LLM API: {str(e)}", exc_info=True)
                self.logger.error(f"Exception type: {type(e)}")
                self.logger.error(f"Exception args: {e.args}")
                raise

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def _get_graph_overview(self, query_text: str) -> Optional[str]:
        """Enhanced graph overview with hybrid retrieval"""
        try:
            if not self.graph:
                return None

            # Extract query entities and keywords using semantic processor
            semantic_analysis = self._semantic_processor.analyze_query(query_text)
            query_entities = [entity['text'].lower() for entity in semantic_analysis['entities']]
            keywords = query_text.lower().split()

            # Enhanced entity-focused query
            entity_query = """
            // Match entities based on name matches
            MATCH (e:Entity)
            WHERE e.name IS NOT NULL
            AND any(keyword IN $keywords WHERE toLower(e.name) CONTAINS toLower(keyword))
            
            // Get connected documents and relationships
            OPTIONAL MATCH (d:Document)-[r:CONTAINS]->(e)
            WHERE d.title IS NOT NULL
            
            // Aggregate results with scoring
            WITH e,
                 collect(DISTINCT {
                   title: d.title,
                   relationship: type(r)
                 }) as document_refs,
                 count(DISTINCT d) as doc_count
            
            RETURN {
              name: e.name,
              type: e.type,
              documents: [doc in document_refs | doc.title],
              relevance: doc_count
            } as entity_info
            ORDER BY entity_info.relevance DESC
            LIMIT 10
            """
            entity_results = self.graph.run(entity_query, 
                                          keywords=keywords, 
                                          entities=[e.lower() for e in query_entities]).data()

            # Enhanced hybrid retrieval combining semantic and graph structure
            doc_query = """
            MATCH (d:Document)-[r:CONTAINS]->(e:Entity)
            WHERE any(keyword IN $keywords WHERE 
                  toLower(d.title) CONTAINS keyword OR
                  toLower(d.content) CONTAINS keyword)
            OR e.name IN $entities
            WITH d {.title, .content} as doc_info,
                 d.embedding as doc_embedding,
                 $embedding as query_embedding,
                 count(distinct e) as entity_matches
            WITH doc_info, doc_embedding, query_embedding, entity_matches,
                 CASE 
                    WHEN doc_embedding IS NOT NULL
                    THEN reduce(dot = 0.0, i IN range(0, size(doc_embedding)-1) | 
                         dot + doc_embedding[i] * query_embedding[i]) /
                         (sqrt(reduce(norm = 0.0, i IN range(0, size(doc_embedding)-1) | 
                         norm + doc_embedding[i] * doc_embedding[i])) *
                         sqrt(reduce(norm = 0.0, i IN range(0, size(query_embedding)-1) | 
                         norm + query_embedding[i] * query_embedding[i])))
                    ELSE 0.0
                 END as semantic_score
            WITH doc_info, entity_matches,
                 semantic_score * 0.6 + 
                 CASE WHEN entity_matches > 0 
                 THEN 0.4 * (entity_matches/5.0) ELSE 0 END as combined_score
            WHERE combined_score > 0.3
            RETURN doc_info, combined_score, entity_matches
            ORDER BY combined_score DESC
            LIMIT 5
            """
            doc_results = self.graph.run(doc_query, embedding=self._semantic_processor.get_text_embedding(query_text)).data()


            if not entity_results and not doc_results:
                return None

            # Format overview
            overview = []

            # Add document information
            if doc_results:
                overview.append(f"Documents:")
                for result in doc_results:
                    overview.append(f"- {result['doc_info']['title']}")
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