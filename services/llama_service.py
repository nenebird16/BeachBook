from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jGraphStore
import logging
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from urllib.parse import urlparse
from py2neo import Graph, ConnectionProfile
from anthropic import Anthropic
from services.semantic_processor import SemanticProcessor
import json

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = None  # We won't be using LlamaIndex's LLM features

        # Initialize Anthropic client for Claude
        self.anthropic = Anthropic()

        # Initialize semantic processor
        self.semantic_processor = SemanticProcessor()

        try:
            if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
                raise ValueError("Neo4j credentials not properly configured")

            # Parse URI for AuraDB
            uri = urlparse(NEO4J_URI)
            self.logger.debug(f"Original URI scheme: {uri.scheme}")
            self.logger.debug(f"Original URI netloc: {uri.netloc}")

            # Initialize direct Neo4j connection for queries
            profile = ConnectionProfile(
                scheme="bolt+s" if uri.scheme == 'neo4j+s' else uri.scheme,
                host=uri.netloc,
                port=7687,
                secure=True if uri.scheme == 'neo4j+s' else False,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            self.graph = Graph(profile=profile)
            self.logger.info("Successfully connected to Neo4j database")

            # Initialize graph store
            self.graph_store = Neo4jGraphStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=f"bolt+s://{uri.netloc}" if uri.scheme == 'neo4j+s' else NEO4J_URI,
                database="neo4j"
            )
            self.logger.info("Successfully initialized Neo4j graph store")

            # Create vector index if it doesn't exist
            self._ensure_vector_index()

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j connections: {str(e)}")
            raise

    def _ensure_vector_index(self):
        """Create vector index for semantic search if it doesn't exist"""
        try:
            # Check if index exists
            index_query = """
            SHOW INDEXES
            YIELD name, type
            WHERE name = 'document_embeddings'
            """
            result = list(self.graph.run(index_query))

            if not result:
                # Create vector index
                create_index = """
                CALL db.index.vector.createNodeIndex(
                    'document_embeddings',
                    'Document',
                    'embedding',
                    1536,
                    'cosine'
                )
                """
                self.graph.run(create_index)
                self.logger.info("Created vector index for document embeddings")
            else:
                self.logger.info("Vector index already exists")

        except Exception as e:
            self.logger.error(f"Error ensuring vector index: {str(e)}")
            raise

    def process_document(self, content):
        """Process document content for storage"""
        try:
            self.logger.info("Processing document for storage")
            # Process document with semantic processor
            doc_info = self.semantic_processor.process_document(content)

            # Store embeddings in Neo4j
            for chunk in doc_info['embeddings']:
                query = """
                MATCH (d:Document {content: $content})
                SET d.embedding = $embedding
                """
                self.graph.run(query, content=chunk['text'], 
                             embedding=chunk['embedding'])

            return True

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def generate_response(self, query, context_info=None):
        """Generate a natural language response using Claude"""
        try:
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
                prompt = f"""As a knowledge graph assistant, I need to respond to this query: "{query}"

                Since I don't find any matches in the knowledge graph for this query, I should:
                1. Politely explain that I can only provide information that exists in the knowledge graph
                2. Suggest that the user ask about specific topics or documents that might be in the knowledge graph
                3. Avoid engaging in general conversation or discussing topics not present in the graph
                4. Keep the response brief and focused

                For example, if it's a greeting or general chat, respond with something like:
                "I can help you explore information stored in the knowledge graph. While I don't have information about [their query], 
                feel free to ask about specific documents or topics in the knowledge base."

                Response:"""

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

            return response.content[0].text

        except Exception as e:
            self.logger.error(f"Error generating Claude response: {str(e)}")
            return None

    def process_query(self, query_text):
        """Process a query using hybrid search"""
        try:
            self.logger.info(f"Processing query: {query_text}")

            # Analyze query semantically
            query_analysis = self.semantic_processor.analyze_query(query_text)
            query_embedding = query_analysis['embedding']

            # Vector similarity search
            vector_query = """
            CALL db.index.vector.queryNodes(
                'document_embeddings',
                5,
                $embedding
            ) YIELD node, score
            WITH node, score
            MATCH (node)-[:CONTAINS]->(e:Entity)
            RETURN node.content as content,
                   node.title as title,
                   collect(distinct e.name) as entities,
                   score as relevance
            ORDER BY relevance DESC
            """
            vector_results = self.graph.run(vector_query, 
                                          embedding=query_embedding).data()
            self.logger.debug(f"Vector query results: {vector_results}")

            # Content-based search as backup
            content_query = """
            MATCH (d:Document)
            WHERE toLower(d.content) CONTAINS toLower($query)
            MATCH (d)-[r:CONTAINS]->(e:Entity)
            RETURN d.content as content, 
                   d.title as title,
                   collect(distinct e.name) as entities,
                   count(e) as relevance
            ORDER BY relevance DESC
            LIMIT 5
            """
            content_results = self.graph.run(content_query, 
                                           query=query_text).data()
            self.logger.debug(f"Content query results: {content_results}")

            # Entity-based expansion
            entity_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
            WITH e
            MATCH (d:Document)-[:CONTAINS]->(e)
            RETURN d.content as content,
                   d.title as title,
                   collect(distinct e.name) as entities
            LIMIT 5
            """
            entity_results = self.graph.run(entity_query, 
                                          query=query_text).data()
            self.logger.debug(f"Entity query results: {entity_results}")

            # Combine and deduplicate results
            all_results = vector_results + content_results + entity_results
            seen_titles = set()
            unique_results = []
            for result in all_results:
                if result['title'] not in seen_titles:
                    seen_titles.add(result['title'])
                    unique_results.append(result)

            # Prepare context for AI response
            context_info = None
            if unique_results:
                context_sections = []
                for result in unique_results[:5]:  # Top 5 unique results
                    context_sections.append(f"- Document: {result['title']}")
                    context_sections.append(f"  Content: {result['content'][:500]}")
                    if result.get('entities'):
                        context_sections.append(
                            f"  Related concepts: {', '.join(result['entities'])}")
                    context_sections.append("")
                context_info = "\n".join(context_sections)

            # Generate AI response (with or without context)
            ai_response = self.generate_response(query_text, context_info)

            # Return structured response
            return {
                'chat_response': ai_response,
                'queries': {
                    'vector_query': vector_query,
                    'content_query': content_query,
                    'entity_query': entity_query,
                    'query_analysis': query_analysis
                },
                'results': context_info if context_info else "No matches found in knowledge graph"
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise