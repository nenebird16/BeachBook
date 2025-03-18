"""Query templates for the volleyball knowledge graph"""
import logging
from typing import Dict, List, Optional

class QueryTemplates:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Basic query templates
        self.SKILL_QUERIES = {
            "skill_drills": """
            MATCH (s:Skill {name: $skill_name})
            MATCH (d:Drill)-[r:DEVELOPS]->(s)
            OPTIONAL MATCH (d)-[:FOCUSES_ON]->(v:VisualElement)
            RETURN s.name as skill,
                   collect(distinct d.name) as drills,
                   collect(distinct v.name) as visual_elements
            """,
            "skill_prerequisites": """
            MATCH (s:Skill {name: $skill_name})
            MATCH (s)-[r:REQUIRES]->(prereq:Skill)
            RETURN s.name as skill,
                   collect({name: prereq.name, strength: r.strength}) as prerequisites
            """
        }
        
        self.DRILL_QUERIES = {
            "drill_details": """
            MATCH (d:Drill {name: $drill_name})
            MATCH (d)-[:DEVELOPS]->(s:Skill)
            OPTIONAL MATCH (d)-[:FOCUSES_ON]->(v:VisualElement)
            RETURN d.name as drill,
                   d.description as description,
                   d.duration as duration,
                   collect(distinct s.name) as skills,
                   collect(distinct v.name) as visual_elements
            """,
            "drill_progression": """
            MATCH (d:Drill)-[:DEVELOPS]->(s:Skill)
            WITH d, collect(s) as skills
            MATCH (d)-[:FOCUSES_ON]->(v:VisualElement)
            RETURN d.name as drill,
                   [skill IN skills | skill.name] as developed_skills,
                   collect(v.name) as visual_focus
            ORDER BY size(developed_skills) DESC
            """
        }
        
        self.VISUAL_QUERIES = {
            "visual_element_usage": """
            MATCH (v:VisualElement {name: $element_name})
            MATCH (d:Drill)-[:FOCUSES_ON]->(v)
            OPTIONAL MATCH (d)-[:DEVELOPS]->(s:Skill)
            RETURN v.name as visual_element,
                   collect(distinct {
                       drill: d.name,
                       skills: collect(distinct s.name)
                   }) as drill_contexts
            """,
            "skill_visual_requirements": """
            MATCH (s:Skill)-[:REQUIRES_VISUAL]->(v:VisualElement)
            RETURN s.name as skill,
                   collect(v.name) as visual_requirements
            ORDER BY size(visual_requirements) DESC
            """
        }
        
        self.RELATIONSHIP_QUERIES = {
            "skill_network": """
            MATCH (s1:Skill)-[r:REQUIRES]->(s2:Skill)
            RETURN s1.name as skill,
                   collect({
                       target: s2.name,
                       strength: r.strength,
                       transfer: r.transfer_effect
                   }) as relationships
            """,
            "visual_skill_connections": """
            MATCH (v:VisualElement)<-[:FOCUSES_ON]-(d:Drill)-[:DEVELOPS]->(s:Skill)
            RETURN v.name as visual_element,
                   collect(distinct s.name) as connected_skills,
                   count(distinct d) as drill_count
            ORDER BY drill_count DESC
            """
        }

    def get_query(self, category: str, query_name: str) -> Optional[str]:
        """Get a specific query template by category and name"""
        try:
            query_dict = getattr(self, f"{category.upper()}_QUERIES")
            return query_dict.get(query_name)
        except AttributeError:
            self.logger.error(f"Query category {category} not found")
            return None

    def list_available_queries(self) -> Dict[str, List[str]]:
        """List all available query categories and names"""
        categories = {}
        for attr_name in dir(self):
            if attr_name.endswith('_QUERIES'):
                category = attr_name.replace('_QUERIES', '').lower()
                queries = getattr(self, attr_name)
                categories[category] = list(queries.keys())
        return categories
