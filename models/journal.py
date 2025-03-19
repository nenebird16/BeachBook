from datetime import datetime
import logging
from services.graph_service import GraphService

logger = logging.getLogger(__name__)

class JournalEntry:
    """Model for managing journal entries in Neo4j"""

    @staticmethod
    def create_audio_entry(audio_path):
        """Create a new audio journal entry"""
        try:
            graph = GraphService().graph

            # Create journal entry node
            query = """
            CREATE (j:JournalEntry {
                type: 'audio',
                audio_url: $audio_path,
                timestamp: datetime(),
                entry_type: 'audio'
            })
            RETURN j, id(j) as id
            """

            result = graph.run(query, audio_path=audio_path).data()
            if result:
                entry = result[0]['j']
                entry['id'] = result[0]['id']
                return entry
            raise Exception("Failed to create audio entry")

        except Exception as e:
            logger.error(f"Error creating audio entry: {str(e)}")
            raise

    @staticmethod
    def create_text_entry(text):
        """Create a new text journal entry"""
        try:
            graph = GraphService().graph

            # Create journal entry node
            query = """
            CREATE (j:JournalEntry {
                type: 'text',
                content: $text,
                timestamp: datetime(),
                entry_type: 'text'
            })
            RETURN j, id(j) as id
            """

            result = graph.run(query, text=text).data()
            if result:
                entry = result[0]['j']
                entry['id'] = result[0]['id']
                return entry
            raise Exception("Failed to create text entry")

        except Exception as e:
            logger.error(f"Error creating text entry: {str(e)}")
            raise

    @staticmethod
    def get_recent_entries(limit=20):
        """Get recent journal entries"""
        try:
            graph = GraphService().graph

            query = """
            MATCH (j:JournalEntry)
            RETURN j.type as type,
                   j.content as content,
                   j.audio_url as audio_url,
                   j.timestamp as timestamp,
                   id(j) as id
            ORDER BY j.timestamp DESC
            LIMIT $limit
            """

            results = graph.run(query, limit=limit).data()

            # Format entries for JSON response
            entries = []
            for entry in results:
                formatted_entry = {
                    'type': entry['type'],
                    'timestamp': entry['timestamp'].isoformat() if entry['timestamp'] else None,
                    'id': entry['id']
                }

                if entry['type'] == 'text':
                    formatted_entry['content'] = entry['content']
                else:
                    formatted_entry['audio_url'] = entry['audio_url']

                entries.append(formatted_entry)

            return entries

        except Exception as e:
            logger.error(f"Error fetching journal entries: {str(e)}")
            raise