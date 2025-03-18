from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.composability import QueryEngineTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import logging
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from urllib.parse import urlparse
from py2neo import Graph, ConnectionProfile, Node, Relationship
from anthropic import Anthropic
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Settings.llm_api_key = None  # We won't be using LlamaIndex's LLM features

        # Initialize Anthropic client for Claude
        self.anthropic = Anthropic()

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
            
            # Set up query engines
            self._setup_query_engines()

        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j connections: {str(e)}")
            raise

    def _setup_query_engines(self):
        """Set up different query engines for various RAG strategies"""
        # Set up KG RAG retriever
        self.kg_retriever = KnowledgeGraphRAGRetriever(
            storage_context=self.graph_store,
            entity_extract_fn=self._extract_entities_from_query,  # Custom entity extraction
            verbose=True
        )
        
        # Set up hybrid search query engine
        self.kg_query_engine = RetrieverQueryEngine.from_args(
            self.kg_retriever,
            service_context=ServiceContext.from_defaults()
        )
        
        # Create query engine tools for different scenarios
        self.query_engine_tools = [
            QueryEngineTool(
                query_engine=self.kg_query_engine,
                metadata=ToolMetadata(
                    name="kg_volleyball_rag",
                    description="Knowledge graph RAG for beach volleyball domain knowledge"
                )
            )
            # Additional specialized query engines would be added here
        ]

    def _extract_entities_from_query(self, query_text: str) -> List[str]:
        """Extract entities from query text specific to volleyball domain"""
        # This could be enhanced with a domain-specific NER model
        entities = []
        
        # Simple volleyball domain entity extraction
        volleyball_terms = [
            # Skills
            "setting", "passing", "blocking", "serving", "attacking", "digging",
            # Positions
            "blocker", "defender", "setter", "hitter",
            # Visual elements
            "ball tracking", "peripheral vision", "trajectory prediction",
            # Training concepts
            "visual-motor integration", "constraint-led approach"
        ]
        
        for term in volleyball_terms:
            if term.lower() in query_text.lower():
                entities.append(term)
        
        return entities

    def process_document(self, content: str):
        """Process document content for KG and vector storage"""
        try:
            self.logger.info("Processing document for knowledge graph storage")
            
            # Create LlamaIndex document
            doc = Document(text=content)
            
            # Index the document
            index = VectorStoreIndex.from_documents([doc])
            
            # Store metadata
            metadata = {
                "indexed_at": datetime.now().isoformat(),
                "doc_id": doc.doc_id
            }
            
            # Store the index for future queries
            # Note: In a real implementation, you'd persist this
            
            return True
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def generate_response(self, query: str, context_info: Optional[str] = None):
        """Generate a natural language response using Claude"""
        try:
            if context_info:
                prompt = f"""Based on the following context from a beach volleyball knowledge graph, help me answer this query: "{query}"

                Context information:
                {context_info}

                Please provide a natural, conversational response that:
                1. Directly answers the query using the context provided
                2. Incorporates relevant information from the context
                3. Highlights key relationships between volleyball concepts
                4. Connects the concepts to visual-motor integration frameworks if relevant
                5. Suggests related volleyball skills, drills, or concepts to explore
                
                Focus on information from the knowledge graph only.

                Response:"""
            else:
                prompt = f"""As a beach volleyball knowledge graph assistant, I need to respond to this query: "{query}"

                Since I don't find any specific matches in the knowledge graph for this query, I should:
                1. Politely explain that I can only provide information that exists in the beach volleyball knowledge graph
                2. Suggest that the user ask about beach volleyball skills, drills, practice plans, or visual-motor integration concepts
                3. Avoid engaging in general conversation or discussing topics not present in the graph
                4. Keep the response brief and focused

                Response:"""

            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {
                        "role": "assistant",
                        "content": "I am a beach volleyball knowledge graph assistant that provides information from our connected graph database focusing on skills, drills, visual-motor integration, and practice planning. I'll help you explore connections between beach volleyball concepts."
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

    def process_query(self, query_text: str) -> Dict:
        """Process a query using hybrid GraphRAG approach"""
        try:
            self.logger.info(f"Processing query: {query_text}")
            
            # Step 1: Execute knowledge graph queries
            kg_results = self._execute_knowledge_graph_queries(query_text)
            
            # Step 2: Execute vector similarity search
            vector_results = self._execute_vector_search(query_text)
            
            # Step 3: Execute custom volleyball domain queries
            domain_results = self._execute_domain_specific_queries(query_text)
            
            # Step 4: Combine results with ranking
            combined_results = self._combine_results(kg_results, vector_results, domain_results)
            
            # Step 5: Prepare context for AI response
            context_info = self._prepare_context(combined_results)
            
            # Step 6: Generate AI response with or without context
            ai_response = self.generate_response(query_text, context_info)

            # Return structured response
            return {
                'chat_response': ai_response,
                'kg_results': kg_results,
                'vector_results': vector_results,
                'domain_results': domain_results,
                'context': context_info if context_info else "No matches found in knowledge graph"
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(f"Query text was: {query_text}")
            raise

    def _execute_knowledge_graph_queries(self, query_text: str) -> List[Dict]:
        """Execute knowledge graph queries based on the query text"""
        results = []
        
        try:
            # 1. Match content in documents
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
            content_results = self.graph.run(content_query, query=query_text).data()
            results.extend(content_results)
            
            # 2. Match entities by name or properties
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
            entity_results = self.graph.run(entity_query, query=query_text).data()
            results.extend(entity_results)
            
            # 3. Match visual elements specifically
            visual_query = """
                MATCH (v:VisualElement)
                WHERE toLower(v.name) CONTAINS toLower($query)
                MATCH (d:Drill)-[r:FOCUSES_ON]->(v)
                MATCH (d)-[dev:DEVELOPS]->(s:Skill)
                RETURN v.name as visual_element,
                       collect(distinct d.name) as drills,
                       collect(distinct s.name) as related_skills
                LIMIT 5
            """
            visual_results = self.graph.run(visual_query, query=query_text).data()
            if visual_results:
                results.extend(visual_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing knowledge graph queries: {str(e)}")
            return []

    def _execute_vector_search(self, query_text: str) -> List[Dict]:
        """Execute vector similarity search"""
        # In a real implementation, this would use vector embeddings
        # For now, return an empty list
        return []

    def _execute_domain_specific_queries(self, query_text: str) -> List[Dict]:
        """Execute domain-specific queries based on volleyball terminology"""
        results = []
        
        # Check for skill development queries
        if any(term in query_text.lower() for term in ["develop", "improve", "practice", "train"]):
            # Extract skills mentioned in the query
            skills = self._extract_skills_from_query(query_text)
            
            if skills:
                for skill in skills:
                    # Find drills that develop this skill
                    drill_query = """
                        MATCH (s:Skill {name: $skill_name})
                        MATCH (d:Drill)-[r:DEVELOPS]->(s)
                        OPTIONAL MATCH (d)-[:FOCUSES_ON]->(v:VisualElement)
                        RETURN s.name as skill,
                               collect(distinct d.name) as recommended_drills,
                               collect(distinct v.name) as visual_elements,
                               count(d) as relevance
                        ORDER BY relevance DESC
                    """
                    drill_results = self.graph.run(drill_query, skill_name=skill).data()
                    if drill_results:
                        results.extend(drill_results)
        
        # Check for practice plan queries
        if any(term in query_text.lower() for term in ["plan", "session", "practice", "workout"]):
            plan_query = """
                MATCH (p:PracticePlan)
                MATCH (p)-[inc:INCLUDES]->(d:Drill)
                RETURN p.name as practice_plan,
                       p.focus as focus,
                       p.duration as duration,
                       collect(distinct d.name) as drills
                LIMIT 3
            """
            plan_results = self.graph.run(plan_query).data()
            if plan_results:
                results.extend(plan_results)
        
        return results

    def _extract_skills_from_query(self, query_text: str) -> List[str]:
        """Extract skill names from the query text"""
        skills = []
        
        # Common beach volleyball skills
        common_skills = [
            "Passing", "Setting", "Hitting", "Blocking", "Serving", 
            "Defense", "Jump Serve", "Float Serve", "Cut Shot", "Line Shot"
        ]
        
        for skill in common_skills:
            if skill.lower() in query_text.lower():
                skills.append(skill)
        
        return skills

    def _combine_results(self, kg_results: List[Dict], vector_results: List[Dict], 
                        domain_results: List[Dict]) -> List[Dict]:
        """Combine and rank results from different retrieval methods"""
        # Simple combination for now - could be enhanced with proper ranking
        combined = []
        
        # Add domain-specific results first (highest priority)
        combined.extend(domain_results)
        
        # Add knowledge graph results
        combined.extend(kg_results)
        
        # Add vector results
        combined.extend(vector_results)
        
        # Remove duplicates (based on simple title comparison)
        seen_titles = set()
        unique_results = []
        
        for result in combined:
            title = result.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(result)
            elif not title:  # Handle results without title
                unique_results.append(result)
        
        return unique_results

    def _prepare_context(self, results: List[Dict]) -> Optional[str]:
        """Prepare context for AI response from combined results"""
        if not results:
            return None
            
        context_sections = []
        
        # Group results by type
        skill_results = [r for r in results if 'skill' in r]
        drill_results = [r for r in results if 'recommended_drills' in r or 'drills' in r]
        visual_results = [r for r in results if 'visual_element' in r or 'visual_elements' in r]
        doc_results = [r for r in results if 'content' in r]
        practice_plan_results = [r for r in results if 'practice_plan' in r]
        
        # Add skill development information
        if skill_results:
            context_sections.append("## Skill Development Information")
            for result in skill_results:
                context_sections.append(f"- Skill: {result.get('skill', 'Unknown')}")
                if 'recommended_drills' in result:
                    drills = result['recommended_drills']
                    context_sections.append(f"  Recommended drills: {', '.join(drills[:3])}" + 
                                         (f" and {len(drills) - 3} more" if len(drills) > 3 else ""))
                if 'visual_elements' in result and result['visual_elements']:
                    context_sections.append(f"  Visual elements: {', '.join(result['visual_elements'])}")
                context_sections.append("")
        
        # Add visual element information
        if visual_results:
            context_sections.append("## Visual Elements")
            for result in visual_results:
                element = result.get('visual_element', '')
                if not element and 'visual_elements' in result:
                    elements = result['visual_elements']
                    for element in elements:
                        context_sections.append(f"- Visual Element: {element}")
                        if 'drills' in result:
                            context_sections.append(f"  Related drills: {', '.join(result['drills'][:3])}")
                        if 'related_skills' in result:
                            context_sections.append(f"  Related skills: {', '.join(result['related_skills'][:3])}")
                        context_sections.append("")
                else:
                    context_sections.append(f"- Visual Element: {element}")
                    if 'drills' in result:
                        context_sections.append(f"  Related drills: {', '.join(result['drills'][:3])}")
                    if 'related_skills' in result:
                        context_sections.append(f"  Related skills: {', '.join(result['related_skills'][:3])}")
                    context_sections.append("")
        
        # Add practice plan information
        if practice_plan_results:
            context_sections.append("## Practice Plans")
            for result in practice_plan_results:
                context_sections.append(f"- Practice Plan: {result.get('practice_plan', 'Unknown')}")
                context_sections.append(f"  Focus: {result.get('focus', 'Not specified')}")
                context_sections.append(f"  Duration: {result.get('duration', 'Not specified')} minutes")
                if 'drills' in result:
                    context_sections.append(f"  Includes drills: {', '.join(result['drills'][:3])}" +
                                         (f" and {len(result['drills']) - 3} more" if len(result['drills']) > 3 else ""))
                context_sections.append("")
        
        # Add document content excerpts
        if doc_results:
            context_sections.append("## Related Documents")
            for i, result in enumerate(doc_results[:3]):
                context_sections.append(f"- Document: {result.get('title', f'Document {i+1}')}")
                # Get first 200 chars of content as excerpt
                content = result.get('content', '')
                excerpt = content[:200] + "..." if len(content) > 200 else content
                context_sections.append(f"  Excerpt: {excerpt}")
                if 'entities' in result:
                    context_sections.append(f"  Related concepts: {', '.join(result['entities'][:5])}")
                context_sections.append("")
        
        return "\n".join(context_sections)

    def execute_cypher_query(self, query_text: str, params: Dict = None) -> List[Dict]:
        """Execute a custom Cypher query directly"""
        try:
            if params is None:
                params = {}
                
            results = self.graph.run(query_text, **params).data()
            return results
        except Exception as e:
            self.logger.error(f"Error executing Cypher query: {str(e)}")
            self.logger.error(f"Query: {query_text}")
            self.logger.error(f"Params: {params}")
            raise

    def get_graph_schema(self) -> Dict:
        """Retrieve the current schema of the graph database"""
        try:
            # Get node labels
            label_query = "CALL db.labels()"
            labels = [record["label"] for record in self.graph.run(label_query).data()]
            
            # Get relationship types
            rel_query = "CALL db.relationshipTypes()"
            relationships = [record["relationshipType"] for record in self.graph.run(rel_query).data()]
            
            # Get property keys
            prop_query = "CALL db.propertyKeys()"
            properties = [record["propertyKey"] for record in self.graph.run(prop_query).data()]
            
            # Get node counts by label
            node_counts = {}
            for label in labels:
                count_query = f"MATCH (n:{label}) RETURN count(n) as count"
                count = self.graph.run(count_query).data()[0]["count"]
                node_counts[label] = count
            
            # Get relationship counts by type
            rel_counts = {}
            for rel_type in relationships:
                count_query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                count = self.graph.run(count_query).data()[0]["count"]
                rel_counts[rel_type] = count
            
            return {
                "labels": labels,
                "relationships": relationships,
                "properties": properties,
                "node_counts": node_counts,
                "relationship_counts": rel_counts
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving graph schema: {str(e)}")
            raise