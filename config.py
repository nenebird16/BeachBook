import os
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Storage Configuration
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "volleyball-knowledge-base")
UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Neo4j Configuration with validation
def validate_neo4j_uri(uri: str) -> bool:
    """Validate Neo4j URI format"""
    if not uri:
        return False

    try:
        parsed = urlparse(uri)
        valid_schemes = ('neo4j://', 'neo4j+s://', 'bolt://', 'bolt+s://')
        has_valid_scheme = any(uri.startswith(scheme) for scheme in valid_schemes)
        has_host = bool(parsed.netloc)

        if not has_valid_scheme:
            logger.error(f"Invalid Neo4j URI scheme. Must start with one of: {', '.join(valid_schemes)}")
            return False
        if not has_host:
            logger.error("Invalid Neo4j URI: Missing host")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating Neo4j URI: {str(e)}")
        return False

def get_validated_env_var(var_name: str) -> str:
    """Get and validate environment variable"""
    value = os.environ.get(var_name)
    if not value:
        logger.error(f"Required environment variable {var_name} is not set")
        raise ValueError(f"Missing required environment variable: {var_name}")

    if var_name == "NEO4J_URI" and not validate_neo4j_uri(value):
        raise ValueError("Invalid Neo4j URI format")

    return value

try:
    # Neo4j Configuration
    NEO4J_URI = get_validated_env_var("NEO4J_URI")
    NEO4J_USER = get_validated_env_var("NEO4J_USER")
    NEO4J_PASSWORD = get_validated_env_var("NEO4J_PASSWORD")

    # Log configuration status (without exposing sensitive values)
    logger.info("Neo4j configuration loaded successfully")
    logger.debug(f"Neo4j URI configured: {'Yes' if NEO4J_URI else 'No'}")
    logger.debug(f"Neo4j User configured: {'Yes' if NEO4J_USER else 'No'}")
    logger.debug(f"Neo4j Password configured: {'Yes' if NEO4J_PASSWORD else 'No'}")

except ValueError as e:
    logger.error(f"Configuration error: {str(e)}")
    # Don't re-raise, let the application start in degraded mode
    NEO4J_URI = None
    NEO4J_USER = None
    NEO4J_PASSWORD = None

# LlamaIndex Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
logger.debug(f"OpenAI API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")