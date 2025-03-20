import os
import logging
from typing import Dict, List, Any, Optional
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from urllib.parse import urlparse
from py2neo import Graph, ConnectionProfile
from anthropic import Anthropic
from services.semantic_processor import SemanticProcessor
from services.query_templates import QueryTemplates

# Configure logging
logger = logging.getLogger(__name__)

class LlamaService:
    def __init__(self):
        """Initialize the LlamaService with required components"""
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = None  # We won't be using LlamaIndex's LLM features

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

            # Initialize and verify semantic processor
            self.logger.info("Initializing semantic processor...")
            self.semantic_processor = SemanticProcessor()

            # Initialize query templates
            self.query_templates = QueryTemplates()
            self.logger.debug("Query templates initialized")

            # Verify spaCy model and vectors
            if not hasattr(self.semantic_processor, 'nlp'):
                self.logger.error("Semantic processor's spaCy model not initialized")
                raise ValueError("Semantic processor's spaCy model not initialized")

            if not self.semantic_processor.nlp.has_vector:
                self.logger.error("Semantic processor's spaCy model has no word vectors")
                raise ValueError("Semantic processor's spaCy model has no word vectors")

            # Test vector generation
            test_text = "volleyball skills"
            test_embedding = self.semantic_processor.get_text_embedding(test_text)
            if not test_embedding or len(test_embedding) == 0:
                self.logger.error("Failed to generate test embedding")
                raise ValueError("Failed to generate test embedding")

            self.logger.info("Semantic processor initialized and verified successfully")

            # Verify Neo4j credentials
            if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                self.logger.error("Neo4j credentials not properly configured")
                raise ValueError("Neo4j credentials not properly configured")

            # Parse URI and setup Neo4j connection
            uri = urlparse(NEO4J_URI)
            self.logger.debug(f"Neo4j URI scheme: {uri.scheme}")
            self.logger.debug(f"Neo4j URI netloc: {uri.netloc}")

            # Configure connection based on URI scheme
            bolt_uri = f"bolt+s://{uri.netloc}" if uri.scheme == 'neo4j+s' else f"bolt://{uri.netloc}"
            self.logger.info(f"Using connection format: {bolt_uri}")

            # Initialize Neo4j connection
            profile = ConnectionProfile(
                uri=bolt_uri,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            self.graph = Graph(profile=profile)

            # Verify Neo4j connection
            test_result = self.graph.run("RETURN 1 as test").data()
            self.logger.debug(f"Neo4j test query result: {test_result}")
            self.logger.info("Successfully connected to Neo4j database")

            # Initialize graph store for LlamaIndex
            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=bolt_uri,
                database="neo4j"
            )
            self.logger.info("Successfully initialized Neo4j graph store")

        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaService: {str(e)}")
            raise

    def process_query(self, query_text: str) -> Dict[str, Any]:
        """Process a query using hybrid search"""
        try:
            self.logger.info(f"Processing query: {query_text}")

            # Get graph context
            graph_results = self._get_graph_overview()

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

    def _get_graph_overview(self) -> Optional[str]:
        """Get an overview of entities and topics in the graph"""
        try:
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

    def generate_response(self, query: str, context_info: Optional[str] = None) -> str:
        """Generate a natural language response using Claude"""
        try:
            self.logger.debug("Starting response generation")
            self.logger.debug(f"Query: {query}")
            self.logger.debug(f"Context available: {'Yes' if context_info else 'No'}")

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
                    # Get graph overview
                    overview = self._get_graph_overview()
                    if overview:
                        prompt = f"""As a knowledge graph assistant, I need to respond to this query: "{query}"

                        Here's what I found in the knowledge graph:
                        {overview}

                        Please provide a helpful response that:
                        1. Summarizes the types of information available
                        2. Lists some key topics or entities found
                        3. Encourages exploring specific areas of interest
                        4. Maintains a focus on actual graph contents

                        Response:"""
                    else:
                        prompt = """The knowledge graph appears to be empty at the moment. Please explain that:
                        1. No documents or entities have been added yet
                        2. Documents need to be uploaded first
                        3. Keep the response brief and clear
                        Response:"""
                else:
                    prompt = f"""As a knowledge graph assistant, I need to respond to this query: "{query}"

                    Since I don't find any matches in the knowledge graph for this query, I should:
                    1. Politely explain that I can only provide information that exists in the knowledge graph
                    2. Suggest that the user ask about specific topics or documents that might be in the knowledge graph
                    3. Avoid engaging in general conversation or discussing topics not present in the graph
                    4. Keep the response brief and focused

                    Response:"""

            self.logger.debug("Sending request to Anthropic")
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-opus-20240229",
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

    def process_document(self, content: str) -> bool:
        """Process document content for knowledge graph storage"""
        try:
            self.logger.info("Processing document for storage")

            # Process document with semantic processor
            semantic_data = self.semantic_processor.process_document(content)

            # Store embeddings in Neo4j if available
            if semantic_data and 'embeddings' in semantic_data:
                for chunk in semantic_data['embeddings']:
                    try:
                        # We'll match on a substring of the content to avoid length issues
                        content_preview = chunk['text'][:200] if len(chunk['text']) > 200 else chunk['text']
                        query = """
                        MATCH (d:Document)
                        WHERE d.content CONTAINS $content_preview
                        SET d.embedding = $embedding
                        """
                        self.graph.run(query,
                                        content_preview=content_preview,
                                        embedding=chunk['embedding'])
                    except Exception as e:
                        self.logger.error(f"Error storing embedding: {str(e)}")
                        continue

            return True

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def get_available_queries(self) -> Dict:
        """Get information about available query templates"""
        return self.query_templates.list_available_queries()

    def execute_template_query(self, category: str, query_name: str, params: Dict = None) -> list:
        """Execute a template query with parameters"""
        try:
            query = self.query_templates.get_query(category, query_name)
            if not query:
                raise ValueError(f"Query template not found: {category}/{query_name}")

            params = params or {}
            results = self.graph.run(query, **params).data()
            return results

        except Exception as e:
            self.logger.error(f"Error executing template query: {str(e)}")
            raise

    def _find_similar_documents(self, query_embedding, threshold=0.7):
        """Find documents with similar embeddings"""
        query = """
        MATCH (d:Document)
        WHERE d.embedding IS NOT NULL
        WITH d, 
        REDUCE(dot = 0.0, i IN RANGE(0, SIZE($embedding)-1) | 
            dot + d.embedding[i] * $embedding[i]) /
        (SQRT(REDUCE(norm1 = 0.0, i IN RANGE(0, SIZE($embedding)-1) | 
            norm1 + d.embedding[i] * d.embedding[i])) *
        SQRT(REDUCE(norm2 = 0.0, i IN RANGE(0, SIZE($embedding)-1) | 
            norm2 + $embedding[i] * $embedding[i]))) AS score
        WITH d, score
        WHERE d.title CONTAINS 'Create Lasting Adaptations'
        RETURN {
            title: d.title,
            content: d.content,
            score: score
        } as document
        ORDER BY score DESC
        LIMIT 5
        """
        return self.graph.run(query, embedding=query_embedding, threshold=threshold).data()

    def _get_relevant_entities(self, documents):
        """Get entities from relevant documents"""
        doc_titles = [doc['document']['title'] for doc in documents]
        query = """
        MATCH (d:Document)-[:CONTAINS]->(e:Entity)
        WHERE d.title IN $titles
        RETURN DISTINCT e.name, e.type, count(d) as relevance
        ORDER BY relevance DESC
        """
        return self.graph.run(query, titles=doc_titles).data()

    def _get_focused_context(self, query_text, focus, docs, entities):
        """Get focused context based on query intent"""
        context = []
        
        # Add document summaries
        if docs:
            context.append("Relevant documents:")
            for doc in docs[:3]:
                context.append(f"- {doc['title']} (similarity: {doc['score']:.2f})")

        # Add entity information
        if entities:
            context.append("\nKey concepts:")
            for entity in entities[:5]:
                context.append(f"- {entity['name']} ({entity['type']})")

        # Add targeted relationship queries based on focus
        if focus.get('main_noun'):
            related = self._get_related_concepts(focus['main_noun'])
            if related:
                context.append("\nRelated concepts:")
                context.extend(related)

        return "\n".join(context)

    def _get_related_concepts(self, concept):
        """Get concepts related to the main query focus"""
        query = """
        MATCH (e:Entity {name: $concept})-[r]-(related:Entity)
        RETURN type(r) as relationship, related.name, related.type
        LIMIT 5
        """
        results = self.graph.run(query, concept=concept).data()
        return [f"- {r['related.name']} ({r['relationship']}) {r['related.type']}"
                for r in results]