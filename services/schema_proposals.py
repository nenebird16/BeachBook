"""Schema proposals based on BVBontology integration"""

PROPOSED_ENTITY_SCHEMAS = {
    "TechnicalSkill": {
        "required": ["name", "category"],
        "optional": ["description", "difficulty_level", "prerequisites", "visual_requirements"],
        "types": {
            "name": str,
            "category": str,  # e.g., "offensive", "defensive"
            "description": str,
            "difficulty_level": int,  # 1-5 scale
            "prerequisites": list,
            "visual_requirements": list
        }
    },
    "TacticalSkill": {
        "required": ["name", "phase"],
        "optional": ["description", "game_situations", "decision_factors"],
        "types": {
            "name": str,
            "phase": str,  # e.g., "serving", "reception", "transition"
            "description": str,
            "game_situations": list,
            "decision_factors": list
        }
    },
    "MatchPhase": {
        "required": ["name"],
        "optional": ["description", "key_skills", "common_patterns"],
        "types": {
            "name": str,
            "description": str,
            "key_skills": list,
            "common_patterns": list
        }
    },
    "DrillProgression": {
        "required": ["name", "target_skill"],
        "optional": ["difficulty", "prerequisites", "next_progressions"],
        "types": {
            "name": str,
            "target_skill": str,
            "difficulty": int,
            "prerequisites": list,
            "next_progressions": list
        }
    }
}

PROPOSED_RELATIONSHIPS = {
    "PROGRESSES_TO": {
        "source": ["TechnicalSkill", "DrillProgression"],
        "target": ["TechnicalSkill", "DrillProgression"],
        "properties": {
            "required": ["progression_type"],
            "optional": ["difficulty_increase", "key_differences"],
            "types": {
                "progression_type": str,  # e.g., "linear", "branching"
                "difficulty_increase": int,
                "key_differences": list
            }
        }
    },
    "USED_IN": {
        "source": ["TechnicalSkill", "TacticalSkill"],
        "target": ["MatchPhase"],
        "properties": {
            "required": ["importance"],
            "optional": ["frequency", "success_factors"],
            "types": {
                "importance": int,  # 1-5 scale
                "frequency": float,  # 0-1 scale
                "success_factors": list
            }
        }
    },
    "COMBINES_WITH": {
        "source": ["TechnicalSkill"],
        "target": ["TechnicalSkill"],
        "properties": {
            "required": ["combination_type"],
            "optional": ["effectiveness", "common_situations"],
            "types": {
                "combination_type": str,  # e.g., "sequence", "simultaneous"
                "effectiveness": int,
                "common_situations": list
            }
        }
    }
}

EXAMPLE_QUERIES = {
    "skill_progression": """
    MATCH (s:TechnicalSkill)-[r:PROGRESSES_TO]->(next:TechnicalSkill)
    WHERE s.name = $skill_name
    RETURN s.name as current_skill,
           collect({
               name: next.name,
               difficulty: next.difficulty_level,
               key_differences: r.key_differences
           }) as progression_options
    """,
    
    "match_phase_skills": """
    MATCH (mp:MatchPhase {name: $phase_name})
    MATCH (s)-[r:USED_IN]->(mp)
    WHERE s:TechnicalSkill OR s:TacticalSkill
    RETURN mp.name as phase,
           collect({
               skill: s.name,
               type: labels(s)[0],
               importance: r.importance
           }) as required_skills
    ORDER BY r.importance DESC
    """,
    
    "skill_combinations": """
    MATCH (s:TechnicalSkill {name: $skill_name})
    MATCH (s)-[r:COMBINES_WITH]->(other:TechnicalSkill)
    RETURN s.name as base_skill,
           collect({
               skill: other.name,
               type: r.combination_type,
               effectiveness: r.effectiveness,
               situations: r.common_situations
           }) as combinations
    ORDER BY r.effectiveness DESC
    """
}
