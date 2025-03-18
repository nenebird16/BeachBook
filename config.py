import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Neo4j Configuration
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# Log configuration status (without exposing sensitive values)
logger.debug(f"Neo4j URI configured: {'Yes' if NEO4J_URI else 'No'}")
logger.debug(f"Neo4j User configured: {'Yes' if NEO4J_USER else 'No'}")
logger.debug(f"Neo4j Password configured: {'Yes' if NEO4J_PASSWORD else 'No'}")

# LlamaIndex Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
logger.debug(f"OpenAI API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")

# Flask Configuration
UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size