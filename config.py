import os

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# LlamaIndex Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Flask Configuration
UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size